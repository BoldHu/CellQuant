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


def asset(name, digest, dataset=False, cache_name=None):
    from huggingface_hub import hf_hub_download
    path = ROOT/'.cache/output_matching_small'/(cache_name or Path(name).name)
    if not path.is_file():
        path = Path(hf_hub_download(DATA if dataset else MODEL, name, revision=DATA_REV if dataset else MODEL_REV,
                                    repo_type='dataset' if dataset else 'model'))
    if sha256_file(path) != digest:
        raise ValueError(f'Pinned file checksum mismatch: {path}')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--confirm-hybrid',action='store_true',help='Evaluate the fixed grid-then-matching combination on eight untouched shared articles')
    modes.add_argument('--validate-batches',action='store_true',help='Replay the archived 42M batch fits on 32 unused train articles')
    modes.add_argument('--smoke-retest',action='store_true',help='Exercise the complete retest and matched controls with a four-pair schedule on 42M')
    modes.add_argument('--screen-bilateral',action='store_true',help='Test input-aware angle proposals on the archived 42M fit panels')
    modes.add_argument('--validate-bilateral',action='store_true',help='Score the frozen 42M input-aware proposals on untouched train articles')
    modes.add_argument('--diagnose-bilateral-rows',action='store_true',help='Verify the row diagnostic against the frozen 42M document results')
    modes.add_argument('--crossfit-rows',action='store_true',help='Swap the first up projection calibration rows on pretrained 42M')
    args=parser.parse_args()
    document_mode = args.validate_bilateral or args.diagnose_bilateral_rows
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
    prefix='batch_retest_42m_' if args.smoke_retest else 'batch_documents_42m_' if args.validate_batches else 'matching_42m_'
    if args.screen_bilateral:
        prefix='bilateral_fit_42m_'
    if document_mode:
        prefix='bilateral_documents_42m_'
    if args.diagnose_bilateral_rows:
        prefix='bilateral_rows_42m_'
    if args.crossfit_rows:
        prefix='row_crossfit_42m_'
    out=ROOT/'runs/experiments'/(prefix+uuid4().hex[:12])
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
    study_split='validation'
    if args.validate_batches or args.smoke_retest or document_mode or args.crossfit_rows:
        from cellquant.batch_validation import DESIGN as batch_design, measure_frozen_batches
        from cellquant.matching_batches import load_matching_arms
        from cellquant.matching_study import CONTINUED
        fit_source=ROOT/'evidence/output_matching_20261119/fit_batches_42m.json'
        reference=ROOT/'evidence/output_matching_20261118/hybrid_42m/measurements.zip'
        screen=read_json(fit_source)
        if screen['archive_sha256']!=sha256_file(reference) or screen['next_step']!='fresh_document_pilot':
            raise ValueError('42M batch fits do not match the archived rotations')
        arms=load_matching_arms(reference,cfg['seeds'])
        traces={int(s):t for s,t in screen['by_seed'].items()}
        if any(arms[s][CONTINUED]['fit_documents']!=sorted(b['document_id'] for b in original[s]['fit']) for s in traces):
            raise ValueError('42M fit documents do not match the archive')
        train_hash='e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7'
        train=asset('wikitext-2-raw-v1/train-00000-of-00001.parquet',train_hash,dataset=True,cache_name='train.parquet')
        train_blocks=tokenize_documents(wikitext_documents(parquet.read_table(train,columns=['text']).to_pylist()),tok,256)
        final=confirmation_partitions(train_blocks,original,batch_design['documents_per_seed'],
                                      namespace=batch_design['partition_namespace'],exclude=blocks)
        design=dict(batch_design)
        study_split='train'
        cfg['formal_pilot'].update(heldout_sequences=design['documents_per_seed'],
                                   heldout_tokens_per_panel=design['documents_per_seed']*design['tokens_per_document'])
        cfg['batch_document_pilot']={'design':design, 'scope':'pretrained 42M engineering verification',
                                    'reference_sha256':sha256_file(reference),'fit_source_sha256':sha256_file(fit_source)}
        if args.smoke_retest:
            from cellquant.batch_retest import DESIGN as retest_design, retest_partitions, run_retest
            excluded = [*blocks, *(b for _, dev in final.values() for b in dev)]
            final = retest_partitions(train_blocks,original,excluded)
            del cfg['batch_document_pilot']
            design = {**retest_design,'prefix_pairs':2,'extra_grid_pairs':2,'parent_grid_points':17,'batch_size':2}
            cfg['batch_g1_retest']={'design':design,'scope':'pretrained 42M engineering verification','matching_pairs':4}
            cfg['data']['calibration_split']='train'
        if document_mode or args.crossfit_rows:
            import json
            from zipfile import ZipFile
            from cellquant.batch_retest import retest_partitions
            from cellquant.bilateral_screen import DOCUMENT_DESIGN, score_frozen
            old = [*blocks, *(b for _, held in final.values() for b in held)]
            earlier = retest_partitions(train_blocks, original, old)
            old.extend(b for splits in earlier.values() for articles in splits for b in articles)
            a100 = ROOT/'evidence/output_matching_20261119/batch_g1_retest_7b.zip'
            with ZipFile(a100) as archive:
                old.extend(json.loads(archive.read('inputs/prior_documents.json')))
                for seed in cfg['seeds']:
                    parts = json.loads(archive.read(f'inputs/seed_{seed}/data_manifest.json'))['partitions']
                    old.extend(b for docs in parts.values() for b in docs)
            design = dict(DOCUMENT_DESIGN)
            final = confirmation_partitions(train_blocks, original, design['documents_per_seed'],
                                            namespace=design['partition_namespace'], exclude=old)
            fit_archive = ROOT/'evidence/output_matching_20261119/bilateral_fit_42m.zip'
            del cfg['batch_document_pilot']
            cfg['bilateral_documents'] = {'design':design,'fit_archive_sha256':sha256_file(fit_archive),
                                           'excluded_7b_archive_sha256':sha256_file(a100)}
            if args.crossfit_rows:
                from cellquant.bilateral_documents import COHORT_FILE as past_cohort
                from cellquant.row_crossfit import COHORT_FILE, DESIGN as swap_design, run_crossfit
                folder = ROOT/'evidence/output_matching_20261119'
                old.extend(b for _, docs in final.values() for b in docs)
                for filename in (past_cohort,COHORT_FILE):
                    cohort = read_json(folder/filename)
                    old.extend(b for docs in cohort['heldout'].values() for b in docs)
                    old.extend(cohort.get('previous_42m_train_blocks', []))
                design = {**swap_design, 'grid_pairs':32}
                final = confirmation_partitions(train_blocks,original,design['documents_per_seed'],
                                                namespace=design['partition_namespace'],exclude=old)
                del cfg['bilateral_documents']
                cfg['row_crossfit'] = {'design':design,'excluded_7b_cohort_hash':config_hash(read_json(folder/COHORT_FILE))}
    elif args.screen_bilateral:
        from cellquant.bilateral_screen import DESIGN as screen_design, run_screen
        reference=ROOT/'evidence/output_matching_20261118/hybrid_42m/measurements.zip'
        design=dict(screen_design)
        cfg['bilateral_screen']={'design':design,'scope':'pretrained 42M development study','reference_sha256':sha256_file(reference)}
    else:
        cfg['output_matching_study']={'design':design, 'scope':'pretrained 42M development study'}
    for seed in cfg['seeds']:
        identity={'model_id':MODEL,'model_revision':MODEL_REV,'tokenizer_revision':MODEL_REV,'dataset_id':DATA,
                  'dataset_revision':DATA_REV,'source_split':study_split,'parquet_sha256':data_hash,'seed':seed,'context_length':256}
        if args.validate_batches or document_mode or args.crossfit_rows:
            identity.update(fit_split='validation',dev_split='train',fit_parquet_sha256=data_hash,parquet_sha256=train_hash)
        if args.smoke_retest:
            identity.update(fit_split='train',dev_split='train',parquet_sha256=train_hash)
        training,validation=final[seed]
        path=save_token_manifest(out/f'inputs_{seed}',{'fit':training,'dev':validation},identity)
        cfg['data']['token_manifests'][str(seed)]=str(path)
    protocol=atomic_json(out/'protocol.json',{'status':'FROZEN','design':design,'layers':[0,3,7],'context':256,
        'quantization':cfg['quantization'],'fit_split':cfg['data']['calibration_split'],'source_split':study_split,'token_manifests':{
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
    prepared=prepare_pilot_inputs(cfg,model,graph_identity=graph_id,heldout_only=args.validate_batches or document_mode,
                                  fit_only=args.screen_bilateral)
    if args.diagnose_bilateral_rows or args.crossfit_rows:
        full_up_weight = model.model.layers[0].mlp.up_proj.module.weight.detach()
    del model
    torch.cuda.empty_cache()
    def emit(phase,**details):
        if phase=='matching.evaluate':
            print(phase,details,flush=True)
    if args.crossfit_rows:
        result=run_crossfit(cfg,prepared,full_up_weight,out,reference)
    elif document_mode:
        result=score_frozen(cfg,prepared,out,reference,fit_archive,
            document_reference=ROOT/'evidence/output_matching_20261119/bilateral_documents_42m.zip' if args.diagnose_bilateral_rows else None,
            full_up_weight=full_up_weight if args.diagnose_bilateral_rows else None)
    elif args.screen_bilateral:
        result=run_screen(cfg,prepared,out,reference)
    elif args.smoke_retest:
        result=run_retest(cfg,prepared,out,graph_id,all_controls=True,engineering_pairs=4)
    elif args.validate_batches:
        result=measure_frozen_batches(prepared,arms,traces,out,device=device)
        result.update(scope='pretrained 42M engineering verification',fit_source=cfg['batch_document_pilot'])
    else:
        result=run_matching_study(cfg,prepared,out,execution_fingerprint=fingerprint['sha256'],progress=emit)
    atomic_json(out/'result.json',{**result,'model_id':MODEL,'model_revision':MODEL_REV,
                                   'pretrained_parameter_count':pretrained_parameters,
                                   'parameters_after_norm_fusion_billions':prepared['model_parameter_billions'],'graph_identity':graph_id})
    print(result['status'],out,flush=True)


if __name__=='__main__':
    main()
