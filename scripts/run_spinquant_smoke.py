"""Actual tiny random LLaMA FP/GPTQ checks; never real-model gate evidence."""
import argparse
import copy
import os
from pathlib import Path
import torch
from cellquant.runtime import atomic_json,environment_report
from cellquant.spinquant_adapter import (activate_overlay,export_spinquant_rotation,fixed_r2,fuse_norms,
    rotate_model,install_online_graph,configure_quantization,gptq_quantize,compare_logits,measure_rtn_transfer)
from cellquant.solver import hadamard_rotation
from cellquant.data import token_hash


def run(args):
    os.environ['CELLQUANT_FHT_BACKEND']=args.fht_backend
    activate_overlay(args.overlay)
    from transformers import LlamaConfig,LlamaForCausalLM
    torch.manual_seed(11)
    config=LlamaConfig(hidden_size=16,intermediate_size=64,num_hidden_layers=2,num_attention_heads=2,
                       num_key_value_heads=args.kv_heads,vocab_size=64,max_position_embeddings=128,attention_dropout=0.)
    config._attn_implementation='eager'
    model=LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'norm' in name:p.copy_(torch.rand_like(p)+.5)
    device=args.device
    model.to(device);ids=torch.randint(0,64,(1,16),device=device)
    with torch.no_grad():original=model(ids).logits.cpu()
    fuse_norms(model)
    with torch.no_grad():fused=model(ids).logits.cpu()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    q=hadamard_rotation(16,11);r2=fixed_r2(config,37)
    checkpoint=output/'R.bin';ck=export_spinquant_rotation(q,r2,checkpoint,num_layers=2)
    rotate_model(model,checkpoint,device=device)
    quant={'w_bits':4,'a_bits':4,'k_bits':4,'v_bits':4,'w_group_size':-1,'a_group_size':-1,
           'k_group_size':8,'v_group_size':8,'symmetric_w':True,'symmetric_a':True,'symmetric_k':True,'symmetric_v':True,
           'final_weight_mse_clip':True,'gptq_percdamp':.01,'gptq_act_order':False}
    install_online_graph(model,quant,quantized=False)
    with torch.no_grad():rotated=model(ids).logits.cpu()
    fusion_error=compare_logits(original,fused,token_ids=ids);rotation_error=compare_logits(original,rotated,token_ids=ids)
    result={'source_type':'synthetic_math','model':'random_tiny_llama','environment':environment_report(),
            'fht_backend':args.fht_backend,'num_key_value_heads':args.kv_heads,'checkpoint':ck,'fusion':fusion_error,'rotation':rotation_error,
            'extra_online_r1_ops':0,'gptq_status':'NOT_RUN','scope':'Tiny random structure, not pretrained/7B/Blackwell'}
    if any(e['relative_rms']>1e-4 or abs(e['mean_nll_delta'])>1e-4 for e in (fusion_error,rotation_error)):
        result['status']='FAIL';atomic_json(output/'report.json',result);raise AssertionError(result)
    if args.gptq:
        pristine=copy.deepcopy(model).cpu()
        model.cpu();blocks=[torch.randint(0,64,(1,16)) for _ in range(4)]
        gptq_quantize(model,blocks,quant,device=device)
        interrupted=copy.deepcopy(pristine)
        def stop_after_first(index):
            if index==0:raise KeyboardInterrupt('Intentional tiny-model resume validation')
        try:gptq_quantize(interrupted,blocks,quant,device=device,checkpoint_dir=output/'resume_layers',signature='synthetic_math_smoke_v1',on_layer_checkpoint=stop_after_first)
        except KeyboardInterrupt:result['interrupt_observed']=True
        resumed=copy.deepcopy(pristine)
        gptq_quantize(resumed,blocks,quant,device=device,checkpoint_dir=output/'resume_layers',signature='synthetic_math_smoke_v1')
        expected=dict(model.named_parameters());actual=dict(resumed.named_parameters())
        resume_error=max(float((expected[n].cpu()-actual[n].cpu()).abs().max()) for n in expected)
        if resume_error!=0.:raise AssertionError(f'GPTQ resume weights differ: {resume_error}')
        result['gptq_resume_parameter_max_error']=resume_error
        from cellquant.capture import capture_panels
        fake_blocks=[{'tokens':b[0].numpy(),'document_id':str(i),'block_id':f'{i}:0',
                      'start_token':0,'token_ids_sha256':token_hash(b[0].numpy())} for i,b in enumerate(blocks)]
        saved={n:p.detach().cpu().clone() for n,p in pristine.named_parameters()}
        result['rtn_diagnostic']=measure_rtn_transfer(pristine,fake_blocks,quant,output/'fixture_rtn_nll.jsonl',
            metadata={'source_type':'unit_test_fixture'},device=device)
        if any(not torch.equal(saved[n],p.cpu()) for n,p in pristine.named_parameters()):raise AssertionError('RTN diagnostic did not restore exact base weights')
        result['rtn_weight_restore_exact']=True
        captured=capture_panels(pristine,fake_blocks,layers=[0,1],tokens_per_panel=8,input_weight_rows=4,
                                device=device,graph_identity='tiny_fixture',source_type='unit_test_fixture')
        if not all(p.weight.shape==(16,64) for p in captured if p.placement=='output'):raise AssertionError('Output hidden dimension truncated')
        result['streamed_capture_shapes']=[{'placement':p.placement,'W':list(p.weight.shape),'X':list(p.activation.shape)} for p in captured]
        configure_quantization(model,quant);model.to(device)
        with torch.no_grad():logits=model(ids).logits
        if not torch.isfinite(logits).all():raise AssertionError('Nonfinite GPTQ logits')
        loss=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,64),ids[:,1:].reshape(-1))
        result.update(gptq_status='EXECUTED_FINITE',gptq_nll=float(loss),
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),quantization=quant)
    result['status']='PASS_TINY_STRUCTURE_ONLY';atomic_json(output/'report.json',result)
    print({k:result[k] for k in ['status','fusion','rotation','gptq_status']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--overlay',default='runs/spinquant_overlay_v2');p.add_argument('--output',default='runs/spinquant_smoke')
    p.add_argument('--device',default='cpu');p.add_argument('--gptq',action='store_true');p.add_argument('--fht-backend',choices=['extension','torch_reference'],default='extension')
    p.add_argument('--kv-heads',type=int,choices=[1,2],default=2)
    run(p.parse_args())
