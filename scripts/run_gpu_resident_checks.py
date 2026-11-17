"""Actual CUDA engineering checks, without HF downloads or scientific gate claims."""
import argparse
from dataclasses import asdict
import gc
import os
from pathlib import Path
import time

import numpy as np
import torch

from cellquant.runtime import atomic_json,environment_report
from cellquant.solver import CalibrationPanel,SolverConfig,pair_schedule
from cellquant.quantization import QuantSpec
from cellquant import tensor_solver as ts


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; this check must not fall back to CPU')
    os.environ['CELLQUANT_FHT_BACKEND']=args.fht_backend
    from cellquant.spinquant_adapter import (activate_overlay,export_spinquant_rotation,fixed_r2,
        fuse_norms,rotate_model,install_online_graph,compare_logits,panel_quant_specs)
    from cellquant.capture import capture_panels
    from cellquant.data import token_hash
    from transformers import LlamaConfig,LlamaForCausalLM
    activate_overlay(args.overlay);torch.manual_seed(71);torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False
    config=LlamaConfig(hidden_size=16,intermediate_size=64,num_hidden_layers=2,num_attention_heads=2,
        num_key_value_heads=2,vocab_size=64,max_position_embeddings=128,attention_dropout=0.)
    config._attn_implementation='eager'
    model=LlamaForCausalLM(config).eval().requires_grad_(False).cuda()
    ids=torch.randint(0,64,(1,16),device='cuda')
    with torch.inference_mode():reference=model(ids).logits
    fuse_norms(model);r2=fixed_r2(config,37,device='cuda')
    export_spinquant_rotation(torch.eye(16,device='cuda',dtype=torch.float64),r2,output/'R.bin')
    rotation=rotate_model(model,output/'R.bin')
    quant=dict(w_bits=4,a_bits=4,k_bits=4,v_bits=4,w_group_size=-1,a_group_size=-1,k_group_size=8,v_group_size=8,
               symmetric_w=True,symmetric_a=True,symmetric_k=True,symmetric_v=True)
    install_online_graph(model,quant,quantized=False);model.cuda()
    with torch.inference_mode():actual=model(ids).logits
    equivalence=compare_logits(reference,actual)
    if equivalence['relative_rms']>1e-4:raise AssertionError(equivalence)
    blocks=[]
    for i in range(4):
        tokens=np.arange(16,dtype=np.int64)+i
        blocks.append({'tokens':tokens,'block_id':f'fixture:{i}','document_id':str(i),'token_ids_sha256':token_hash(tokens)})
    layer_devices=[]
    handles=[layer.register_forward_pre_hook(lambda mod,inp:layer_devices.append(str(inp[0].device))) for layer in model.model.layers]
    try:
        panels=capture_panels(model,blocks,layers=[0,1],tokens_per_panel=8,input_weight_rows=4,
            device='cuda',graph_identity='unit_test_fixture',source_type='unit_test_fixture',resident=True)
    finally:
        for handle in handles:handle.remove()
    assert all(p.device.type=='cuda' for p in model.parameters())
    assert all(p.weight.is_cuda and p.activation.is_cuda for p in panels)
    specs=panel_quant_specs(model,panels,quant)
    dimensions=[{'id':p.panel_id,'weight_shape':list(p.weight.shape),'activation_shape':list(p.activation.shape),'device':str(p.weight.device)} for p in panels]
    del model,reference,actual;gc.collect();torch.cuda.empty_cache()
    results={};q0=ts.hadamard_rotation(16,11,'cuda')
    for method in ['hadamard_shared','givens_ste_matched','angle_grid_matched','fixed_scale_search_matched','cellquant']:
        print('GPU check:',method,flush=True)
        cfg=SolverConfig(backend='torch',device='cuda',theta_limit=.01,wall_seconds=180,seconds_per_pair=120)
        result=ts.fit_baseline(method,panels,q0,specs,[(0,1)],cfg,ste_steps=4,grid_points=9)
        q=result.pop('q')
        assert q.is_cuda and result['status']=='COMPLETE'
        assert result['final']['loss']<=result['initial']['loss']+1e-10
        assert all(result['final'][k]<=result['limits'][k]+1e-10 for k in ['budget_w','budget_a'])
        results[method]=result
    full=None
    if args.full_dimension:
        del panels,q0,q;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        w=torch.randn((4096,11008),device='cuda',dtype=torch.float32)*.01
        x=torch.randn((11008,2),device='cuda',dtype=torch.float32)
        p=CalibrationPanel('synthetic_down_4096','output',w,x,metadata={'source_type':'synthetic_math'})
        q=ts.hadamard_rotation(4096,17,'cuda');torch.cuda.synchronize();start=time.perf_counter()
        state=ts.PanelState(p,q,QuantSpec(),QuantSpec())
        measured=state.direct((0,1),.001);torch.cuda.synchronize()
        assert bool(torch.isfinite(measured['loss']))
        full={'source_type':'synthetic_math','weight_shape':list(w.shape),'tokens':2,'device':str(state.a.device),
              'initialization_and_one_direct_seconds':time.perf_counter()-start,'nmse':float(measured['loss']),
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'scope':'Full reduction/hidden dimensions; two-token engineering memory/GEMM probe only, not 7B/G1/full event-search throughput'}
    report={'source_type':'synthetic_math','status':'PASS_ENGINEERING_ONLY','environment':environment_report(),
        'all_model_parameters_cuda':True,'layer_forward_devices':sorted(set(layer_devices)),
        'panels':dimensions,'model_fp_equivalence':equivalence,'identity_rotation':rotation,'controls':results,
        'full_dimension_probe':full,'fht_backend':args.fht_backend,
        'scope':'Random tiny model / synthetic panels; no pretrained-model quality or speedup claim; no G1 evidence'}
    atomic_json(output/'report.json',report)
    print({'status':report['status'],'report':str(output/'report.json'),'fp_equivalence':equivalence},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--overlay',default='runs/spinquant_overlay_v2')
    p.add_argument('--fht-backend',choices=['extension','torch_reference'],default='extension')
    p.add_argument('--full-dimension',action='store_true')
    args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError('Choose a new output directory; previous reports are immutable')
    try:run(args)
    except (Exception,KeyboardInterrupt) as exc:
        # Never manufacture a passing report when the requested CUDA check failed.
        atomic_json(Path(args.output)/'failure.json',{'source_type':'synthetic_math','status':'FAILED','reason':f'{type(exc).__name__}: {exc}'})
        raise
