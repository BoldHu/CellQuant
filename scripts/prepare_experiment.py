"""Resolve real HF revisions, sample disjoint train documents, freeze provenance.

Does not download model weights. Formal test tokens require --include-tests,
a frozen protocol and passing G2; scoring still requires a new bound G3.
"""
from pathlib import Path
import argparse
import copy
import os
from cellquant.runtime import atomic_json,read_json,sha256_file,environment_report,config_hash,huggingface_cache_paths
from cellquant.data import (resolve_revision,wikitext_documents,tokenize_documents,split_train_blocks,
                            save_token_manifest,load_token_manifest,confirmation_partitions,verify_disjoint,immutable_revision)
from cellquant.gates import evaluate_file


def cached_matching_model(config, model_id, weight_files, cache):
    from huggingface_hub import try_to_load_from_cache
    def cached(name, revision):
        path = try_to_load_from_cache(model_id, name, revision=revision, cache_dir=str(cache))
        return isinstance(path, str) and Path(path).is_file()
    required = ['config.json', *(row['path'] for row in weight_files)]
    if len(weight_files) > 1:
        required.append('model.safetensors.index.json')
    return (all(cached(name,config['model_revision']) for name in required)
            and all(cached(name,config['tokenizer_revision']) for name in ('tokenizer_config.json','tokenizer.json')))


def prepare(config_path,output,*,include_tests=False,relative_screen=None,matching_study=False):
    config=read_json(config_path);output=Path(output).resolve()
    previous=None
    parent_dataset_fingerprints=set()
    if relative_screen is not None or matching_study:
        from cellquant.objective_screen import verified_screen, validate_screen_config, verify_parent_manifests
        if (config.get('stage')!='G1' or include_tests or config.get('resume_run_dir')
                or config.get('g1_confirmation') or config.get('output_matching_study')
                or (relative_screen is not None and matching_study)):
            raise ValueError('Select one study on the original frozen G1 training documents')
        validate_screen_config(config)
        if matching_study:
            verify_parent_manifests(config)
        else:
            screen_sha=verified_screen(relative_screen,config)
        previous={}
        for seed in config['seeds']:
            manifest,parts=load_token_manifest(config['data']['token_manifests'][str(seed)])
            identity=manifest['identity']
            if (identity['source_split']!='train' or identity['seed']!=seed or
                    identity['dataset_id']!=config['data']['calibration_dataset'] or
                    any(identity[k]!=config[k] for k in ('model_revision','tokenizer_revision','dataset_revision'))):
                raise ValueError('Parent token identity does not match the G1 protocol')
            previous[seed]=parts
            parent_dataset_fingerprints.add(identity['dataset_fingerprint'])
        parent={'parent_config_sha256':config_hash(config),
                'parent_token_manifests':{s:sha256_file(p) for s,p in config['data']['token_manifests'].items()}}
        if matching_study:
            from cellquant.matching_study import DESIGN
            config['output_matching_study']={**parent, 'design':DESIGN}
        else:
            config['g1_confirmation']={**parent,'screen_result_sha256':screen_sha,
                'partition_seed_namespace':20261118,'extra_control':'original_objective_grid'}
            config['solver']['fit_objective']='relative_site_nmse'
        config.pop('result_path',None)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new preparation directory; frozen artifacts are immutable')
    output.mkdir(parents=True,exist_ok=True)
    data=config['data'];model_id=config.get('model_id',config.get('formal_pilot',{}).get('model_id'))
    token=os.environ.get('HF_TOKEN') or None
    cache,dataset_cache=huggingface_cache_paths(config.get('runtime'))
    from transformers import AutoTokenizer
    from datasets import load_dataset
    if matching_study:
        if not all(immutable_revision(config.get(k)) for k in ('model_revision','tokenizer_revision','dataset_revision')):
            raise ValueError('Output matching requires the three pinned revisions from the original G1')
    else:
        config['model_revision']=resolve_revision(model_id,config.get('model_revision'),token=token)
        config['tokenizer_revision']=resolve_revision(model_id,config.get('tokenizer_revision') or config['model_revision'],token=token)
        config['dataset_revision']=resolve_revision(data['calibration_dataset'],config.get('dataset_revision'),repo_type='dataset',token=token)
    # Resolve metadata before opening formal test splits. Do not resolve mutable
    # task/C4 branches at scoring time.
    if config.get('stage')=='full':
        from cellquant.evaluation import TASK_SOURCES,TASK_METRICS,HARNESS_COMMIT
        ev=config['evaluation'];revisions=ev.get('task_dataset_revisions',{})
        ev['task_metric_keys']=dict(TASK_METRICS)
        ev['task_dataset_revisions']={t:resolve_revision(repo,revisions.get(t),repo_type='dataset',token=token) for t,(_,repo) in TASK_SOURCES.items()}
        config['harness_commit']=HARNESS_COMMIT
        ev.setdefault('harness_path','third_party/lm-evaluation-harness')
        c4=ev['c4'];c4['revision']=resolve_revision(c4['dataset_id'],c4.get('revision'),repo_type='dataset',token=token)
    weight_files=[]
    if matching_study:
        parent_source=Path(config['model_source_manifest_path'])
        if sha256_file(parent_source)!=config['model_source_manifest_sha256']:
            raise ValueError('Original model shard metadata changed')
        source=read_json(parent_source)
        if source['model_id']!=model_id or source['revision']!=config['model_revision']:
            raise ValueError('Original model shard identity does not match this study')
        weight_files=source['files']
    else:
        from huggingface_hub import HfApi
        info=HfApi().model_info(model_id,revision=config['model_revision'],files_metadata=True,token=token)
        for sibling in info.siblings:
            if sibling.rfilename.endswith('.safetensors'):
                lfs=sibling.lfs
                if lfs is None:raise ValueError('Expected LFS SHA-256 metadata for model weights')
                weight_files.append({'path':sibling.rfilename,'size_bytes':sibling.size,'sha256':lfs.sha256})
    if not weight_files:raise ValueError('No safetensors weight metadata in pinned HF model')
    model_manifest=atomic_json(output/'model_source_manifest.json',{'model_id':model_id,'revision':config['model_revision'],'files':weight_files})
    config['model_source_manifest_path']=str(model_manifest)
    config['model_source_manifest_sha256']=sha256_file(model_manifest)
    if matching_study and cached_matching_model(config,model_id,weight_files,cache):
        config['runtime']['local_cache_only']=True
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=config['tokenizer_revision'],cache_dir=str(cache),token=token,
                                           local_files_only=matching_study and config['runtime'].get('local_cache_only',False),
                                           add_bos_token=False,add_eos_token=False)
    ds=load_dataset(data['calibration_dataset'],data['dataset_config'],split='train',revision=config['dataset_revision'],cache_dir=str(dataset_cache),token=token)
    if previous is not None and parent_dataset_fingerprints!={ds._fingerprint}:
        raise ValueError('Confirmation dataset fingerprint changed from the parent calibration corpus')
    context=config.get('formal_pilot',{}).get('context_length',data['sequence_length'])
    blocks=tokenize_documents(wikitext_documents(ds),tokenizer,context)
    env=environment_report();dtype=config['runtime']['dtype']
    if dtype=='auto_detect_and_record':
        if not env['cuda'].get('available'):raise ValueError('No GPU for choosing actual model dtype; set explicit dtype for target platform')
        dtype='bfloat16' if env['cuda'].get('bf16_supported') else 'float16'
    config['runtime'].update(dtype=dtype,cache_dir=str(cache),dataset_cache_dir=str(dataset_cache))
    manifests={};formal=config.get('formal_pilot',{})
    fresh=confirmation_partitions(blocks,previous,formal['heldout_sequences']) if previous is not None else None
    if matching_study:
        reserved = [b for _, dev in fresh.values() for b in dev]
        fresh=confirmation_partitions(blocks,previous,formal['heldout_sequences'],
                                      namespace=DESIGN['partition_namespace'],exclude=reserved)
    tests={}
    if include_tests:
        g2=config.get('g2_evidence_path','runs/gates/G2_evidence.json')
        if not Path(g2).exists() or evaluate_file('G2',g2)['status']!='PASS':raise ValueError('Current G2 evidence required before preparing formal test tokens')
        if config.get('protocol_status')!='FROZEN':raise ValueError('Freeze model and evaluation protocol before formal test access')
        eval_config=config['evaluation']
        wiki=load_dataset(data['calibration_dataset'],data['dataset_config'],split='test',revision=config['dataset_revision'],cache_dir=str(dataset_cache),token=token)
        tests['wikitext2_test']=tokenize_documents(wikitext_documents(wiki),tokenizer,context)
        c4=eval_config['c4']
        if not immutable_revision(c4['revision']):raise ValueError('C4 revision must be frozen, not resolved after test access')
        if not c4.get('data_files') or not c4.get('max_documents'):raise ValueError('Frozen C4 shard/subset limit required')
        c4ds=load_dataset(c4['dataset_id'],c4.get('config','en'),split=c4['split'],revision=c4['revision'],
                          data_files=c4['data_files'],streaming=True,token=token)
        documents=((str(i),row['text']) for i,row in enumerate(c4ds.take(c4['max_documents'])))
        tests['c4_test']=tokenize_documents(documents,tokenizer,context,max_blocks=c4['max_blocks'])
        if len(tests['c4_test'])!=c4['max_blocks']:raise ValueError('Insufficient C4 blocks in the frozen subset')
    for seed in config.get('seeds',[config.get('seed',11)]):
        fit,dev=(fresh[seed] if fresh is not None else split_train_blocks(blocks,
                 formal.get('fit_sequences',data['calibration_samples']),formal.get('heldout_sequences',data.get('dev_sequences',16)),seed))
        verify_disjoint(fit,dev)
        identity={'model_id':model_id,'model_revision':config['model_revision'],'tokenizer_revision':config['tokenizer_revision'],
                  'dataset_id':data['calibration_dataset'],'dataset_revision':config['dataset_revision'],'dataset_fingerprint':ds._fingerprint,
                  'seed':seed,'context_length':context,'source_split':'train','formal_test_specs':config.get('evaluation') if include_tests else None}
        partitions={'fit':fit,'dev':dev,**tests}
        if config.get('spinquant_training'):
            train=config['spinquant_training']
            if train.get('stream')!='all_train_documents_excluding_dev':raise ValueError('Full SpinQuant requires the explicit document-disjoint training pool')
            held_ids={b['document_id'] for b in dev};held_hashes={b['document_sha256'] for b in dev}
            pool=[b for b in blocks if b['document_id'] not in held_ids and b['document_sha256'] not in held_hashes]
            verify_disjoint(pool,dev)
            if not pool:raise ValueError('No independent CE train blocks remain after dev exclusion')
            partitions['spinquant_train']=pool
            identity['spinquant_training_stream']='All eligible WikiText train blocks excluding complete dev documents; same fixed stream for stock FP64 and FP32 port'
        manifests[str(seed)]=str(save_token_manifest(output/f'seed_{seed}',partitions,identity))
    data['token_manifests']=manifests
    data['token_manifest_path']=manifests[str(config.get('seed',config.get('seeds',[11])[0]))]
    protocol={'schema_version':'1.0','status':'FROZEN','model_id':model_id,'model_revision':config['model_revision'],
              'tokenizer_revision':config['tokenizer_revision'],'dataset_revision':config['dataset_revision'],
              'quantization':config['quantization'],'rotation':config['rotation'],'runtime_dtype':dtype,'runtime':config['runtime'],
              'data':{k:v for k,v in data.items() if k not in {'token_manifests','token_manifest_path'}},
              'token_manifest_sha256':{s:sha256_file(p) for s,p in manifests.items()},
              'evaluation':config['evaluation'],'solver':config.get('solver'),'spinquant_training':config.get('spinquant_training'),
              'harness_commit':config.get('harness_commit'),
              'tokenization':'per-document nonoverlapping blocks; no BOS; drop short tails',
              'note':'Own matched protocol; do not compare these PPL values to stock concatenated WikiText numbers'}
    if matching_study:
        protocol['output_matching_study']=config['output_matching_study']
    elif previous is not None:
        protocol['g1_confirmation']=config['g1_confirmation']
    protocol_path=atomic_json(output/'protocol.json',protocol)
    config.update(protocol_manifest_path=str(protocol_path),protocol_sha256=sha256_file(protocol_path),protocol_status='FROZEN')
    provenance={'models':{model_id:{'model_revision':config['model_revision'],'tokenizer_revision':config['tokenizer_revision']}},
                'data_manifest_path':data['token_manifest_path'],'data_manifest_sha256':sha256_file(data['token_manifest_path']),
                'protocol_manifest_path':str(protocol_path),'protocol_sha256':sha256_file(protocol_path),'dtype':dtype}
    prov_path=atomic_json(output/'execution_provenance.json',provenance)
    config['execution_provenance_path']=str(prov_path)
    prepared=atomic_json(output/'config.json',config)
    print(f'Prepared immutable config: {prepared}')
    return prepared


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--include-tests',action='store_true')
    p.add_argument('--relative-g1-screen',help='Completed objective-screen run for the single fresh G1 confirmation')
    p.add_argument('--matching-study',action='store_true',help='Freeze a new disjoint train-document cohort for output matching')
    a=p.parse_args();prepare(a.config,a.output,include_tests=a.include_tests,relative_screen=a.relative_g1_screen,matching_study=a.matching_study)
