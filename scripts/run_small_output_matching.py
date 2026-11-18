"""Reproduce the pretrained 42M mechanism study on a small CUDA GPU."""
import os
import argparse
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cellquant.runtime import atomic_json, config_hash, environment_report, read_json, sha256_file


MODEL = 'Xenova/llama2.c-stories42M'
MODEL_REV = 'be9e05cce52ef2a4615026726bddb12639d541f5'
DATA = 'Salesforce/wikitext'
DATA_REV = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def asset(name, digest, dataset=False):
    from huggingface_hub import hf_hub_download
    path = ROOT/'.cache/output_matching_small'/Path(name).name
    if not path.is_file():
        path = Path(hf_hub_download(DATA if dataset else MODEL, name, revision=DATA_REV if dataset else MODEL_REV,
                                    repo_type='dataset' if dataset else 'model'))
    if sha256_file(path) != digest:
        raise ValueError(f'Pinned file checksum mismatch: {path}')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-hybrid',action='store_true',help='Evaluate the fixed grid-then-matching combination on eight untouched shared articles')
    args=parser.parse_args()
    import pyarrow.parquet as parquet
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from cellquant.data import (confirmation_partitions, save_token_manifest, split_train_blocks, tokenize_documents,
                                verify_disjoint, wikitext_documents)
    from cellquant.matching_study import DESIGN, run_matching_study
    from cellquant.pilot import prepare_pilot_inputs
    from cellquant.runner import execution_fingerprint
    from cellquant.spinquant_adapter import activate_overlay, export_spinquant_rotation, fixed_r2, fuse_norms, install_online_graph, rotate_model
    os.environ['CELLQUANT_FHT_BACKEND']='torch_reference'
    cfg=read_json(ROOT/'configs/colab_pilot.json')
    overlay=activate_overlay(ROOT/cfg['spinquant_overlay'])
    out=ROOT/'runs/experiments'/('matching_42m_'+uuid4().hex[:12])
    out.mkdir(parents=True)
    print('Run',out,flush=True)
    device='cuda:0'
    torch.backends.cuda.matmul.allow_tf32=False
    model_config=LlamaConfig.from_json_file(str(asset('config.json','e08b3deb0dc74730c4e15f366a2ab98412f3f9fadf90a972604db79f76b21837')))
    model_config._attn_implementation='eager'
    model=LlamaForCausalLM(model_config)
    weight_path=asset('pytorch_model.bin','ca5d73fb5862fe2732b76c4e308ed343b4db2abbaf1fc1d58bd5b4f29d96d096')
    model.load_state_dict(torch.load(weight_path,map_location='cpu',weights_only=True),strict=True)
    pretrained_parameters=sum(p.numel() for p in model.parameters())
    model.eval().to(device)
    tok=PreTrainedTokenizerFast(tokenizer_file=str(asset('tokenizer.json','6d5e0f8575e3bba8e9dad476cf67b7f4b77c4a4b7b749331dc89c9b1c53b642b')))
    data_hash='204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c'
    corpus=asset('wikitext-2-raw-v1/validation-00000-of-00001.parquet',data_hash,dataset=True)
    blocks=tokenize_documents(wikitext_documents(parquet.read_table(corpus,columns=['text']).to_pylist()),tok,256)
    fit,dev=split_train_blocks(blocks,16,16,2026111802)
    verify_disjoint(fit,dev)
    original={s:{'fit':fit[i*8:(i+1)*8], 'dev':dev[i*8:(i+1)*8]} for i,s in enumerate(cfg['seeds'])}
    final=confirmation_partitions(blocks,original,8,namespace=2026111803)
    if args.confirm_hybrid:
        prior_dev=dev+[b for _,pool in final.values() for b in pool]
        _, fresh=confirmation_partitions(blocks,{0:{'fit':fit,'dev':prior_dev}},8,namespace=2026111804)[0]
        final={s:(training,fresh) for s,(training,_) in final.items()}
    cfg.update(model_id=MODEL, model_revision=MODEL_REV,tokenizer_revision=MODEL_REV,dataset_revision=DATA_REV)
    cfg['formal_pilot'].update(model_id=MODEL,layers=[0,3,7],context_length=256)
    cfg['runtime'].update(dtype='float32',model_device=device,fht_backend='torch_reference')
    head_width=model.config.hidden_size//model.config.num_attention_heads
    cfg['quantization'].update(k_group_size=head_width,v_group_size=head_width)
    cfg['data'].update(calibration_split='validation',sequence_length=256,token_manifests={})
    cfg['solver'].update(pair_budget=16,wall_seconds_per_method=1440)
    design={**DESIGN,'warm_pairs':16,'extra_grid_pairs':16,'sparse_pairs':16}
    if not args.confirm_hybrid:
        design.pop('refine_continued_grid')
        design['version']=2
    cfg['output_matching_study']={'design':design, 'scope':'pretrained 42M development study'}
    for seed in cfg['seeds']:
        identity={'model_id':MODEL,'model_revision':MODEL_REV,'tokenizer_revision':MODEL_REV,'dataset_id':DATA,
                  'dataset_revision':DATA_REV,'source_split':'validation','parquet_sha256':data_hash,'seed':seed}
        training,validation=final[seed]
        path=save_token_manifest(out/f'inputs_{seed}',{'fit':training,'dev':validation},identity)
        cfg['data']['token_manifests'][str(seed)]=str(path)
    protocol=atomic_json(out/'protocol.json',{'status':'FROZEN','design':design,'layers':[0,3,7],'context':256,
        'quantization':cfg['quantization'],'source_split':'validation','token_manifests':{
        s:sha256_file(p) for s,p in cfg['data']['token_manifests'].items()}})
    first=cfg['data']['token_manifests']['11']
    provenance=atomic_json(out/'execution_provenance.json',{'models':{MODEL:{'model_revision':MODEL_REV,'tokenizer_revision':MODEL_REV}},
        'data_manifest_path':first,'data_manifest_sha256':sha256_file(first),'protocol_manifest_path':str(protocol),
        'protocol_sha256':sha256_file(protocol),'dtype':'float32'})
    atomic_json(out/'config.json',cfg)
    fingerprint=execution_fingerprint(ROOT,provenance)
    if fingerprint['status']!='VERIFIED':
        raise ValueError(fingerprint['missing'])
    atomic_json(out/'source_environment.json',{'code':fingerprint,'environment':environment_report()})
    graph_id=config_hash({'overlay_sha256':overlay['overlay_sha256'],'model_revision':MODEL_REV,
                          'compute_dtype':'float32','r2_seed':cfg['rotation']['fixed_r2_seed'],'fht':'torch_reference'})
    r2=fixed_r2(model.config,cfg['rotation']['fixed_r2_seed'],device=device)
    fuse_norms(model)
    export_spinquant_rotation(torch.eye(model.config.hidden_size,device=device),r2,out/'fixed_R.bin')
    rotate_model(model,out/'fixed_R.bin',device=device)
    install_online_graph(model,cfg['quantization'],quantized=False)
    model.to(device)
    prepared=prepare_pilot_inputs(cfg,model,graph_identity=graph_id)
    del model
    torch.cuda.empty_cache()
    def emit(phase,**details):
        if phase=='matching.evaluate':
            print(phase,details,flush=True)
    result=run_matching_study(cfg,prepared,out,execution_fingerprint=fingerprint['sha256'],progress=emit)
    atomic_json(out/'result.json',{**result,'model_id':MODEL,'model_revision':MODEL_REV,
                                   'pretrained_parameter_count':pretrained_parameters,
                                   'parameters_after_norm_fusion_billions':prepared['model_parameter_billions'],'graph_identity':graph_id})
    print(result['status'],out,flush=True)


if __name__=='__main__':
    main()
