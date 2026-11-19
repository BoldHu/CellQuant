"""Resolve real HF revisions, sample disjoint train documents, freeze provenance.

Does not download model weights. Formal test tokens require --include-tests,
a frozen protocol and passing G2; scoring still requires a new bound G3.
"""
from pathlib import Path
import argparse
import copy
import os
import shutil
from zipfile import ZipFile
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


def prepare(config_path,output,*,include_tests=False,relative_screen=None,matching_study=False,
            batch_screen=None,matching_archive=None,retest_pilot=None,bilateral_sources=None,row_crossfit_cohort=None,row_refresh_sources=None):
    output=Path(output).resolve()
    previous=None
    parent_dataset_fingerprints=set()
    reuse_matching_inputs = matching_study or batch_screen is not None or retest_pilot is not None or bilateral_sources is not None
    if row_crossfit_cohort is not None and bilateral_sources is None:
        raise ValueError('The row swap requires the archived fits and their previous document cohort')
    if row_refresh_sources is not None and row_crossfit_cohort is None:
        raise ValueError('Row refresh requires the original row-crossfit cohort')
    if bilateral_sources is not None:
        from cellquant.bilateral_documents import COHORT_FILE, FIT_FILE, GRID_FILE, DESIGN as bilateral_design
        from cellquant.bilateral_documents import block_identity, draw_documents, read_7b_fit
        from cellquant.gates import load_rules
        if (len(bilateral_sources) != 3 or Path(config_path).resolve() != Path(bilateral_sources[0]).resolve()
                or include_tests or relative_screen is not None or matching_study or batch_screen is not None or retest_pilot is not None):
            raise ValueError('Bilateral document evaluation takes its configuration from the completed fit archive')
        screen = read_7b_fit(bilateral_sources[0], bilateral_sources[1])
        config = copy.deepcopy(screen['config'])
        source = config.pop('batch_g1_retest')
        parent = {k:source[k] for k in ('parent_config_sha256','parent_token_manifests')}
        previous = screen['parts']
        parent_dataset_fingerprints = {screen['dataset_fingerprint']}
    else:
        config=read_json(config_path)
    if bilateral_sources is None and (relative_screen is not None or reuse_matching_inputs):
        from cellquant.objective_screen import verified_screen, validate_screen_config, verify_parent_manifests
        if (config.get('stage')!='G1' or include_tests or config.get('resume_run_dir')
                or config.get('g1_confirmation') or config.get('output_matching_study') or config.get('batch_document_pilot')
                or config.get('batch_g1_retest') or config.get('bilateral_document_pilot')
                or sum((relative_screen is not None, matching_study, batch_screen is not None, retest_pilot is not None)) != 1):
            raise ValueError('Select one study on the original frozen G1 training documents')
        validate_screen_config(config)
        if reuse_matching_inputs:
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
        elif relative_screen is not None:
            config['g1_confirmation']={**parent,'screen_result_sha256':screen_sha,
                'partition_seed_namespace':20261118,'extra_control':'original_objective_grid'}
            config['solver']['fit_objective']='relative_site_nmse'
        config.pop('result_path',None)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new preparation directory; frozen artifacts are immutable')
    output.mkdir(parents=True,exist_ok=True)
    if bilateral_sources is not None:
        refs = output/'fit_reference'
        refs.mkdir()
        hashes = {}
        for name, source_path in zip((FIT_FILE, GRID_FILE, COHORT_FILE), bilateral_sources):
            dest = refs/name
            shutil.copyfile(source_path, dest)
            hashes[name] = sha256_file(dest)
        cohort = read_json(refs/COHORT_FILE)
        if (cohort['fit_archive_sha256'] != hashes[FIT_FILE] or screen['receipt']['reference_archive_sha256'] != hashes[GRID_FILE]
                or cohort['design'] != bilateral_design or cohort['dataset_revision'] != config['dataset_revision']):
            raise ValueError('The frozen document list, fit and grid belong to different runs')
        with ZipFile(refs/FIT_FILE) as archive:
            model_source = refs/'model_source_manifest.json'
            model_source.write_bytes(archive.read('inputs/model_source_manifest.json'))
            config['model_source_manifest_path'] = str(model_source)
        config['bilateral_document_pilot'] = {**parent, 'design':bilateral_design, 'archives':hashes, 'quality_rules_sha256':config_hash(load_rules()),
                                               'screen_run_id':screen['receipt']['run_id']}
        if row_crossfit_cohort is not None:
            from cellquant.row_crossfit import COHORT_FILE as crossfit_file, DESIGN as crossfit_design
            dest = refs/crossfit_file
            shutil.copyfile(row_crossfit_cohort,dest)
            new_cohort = read_json(dest)
            if new_cohort['previous_cohort_hash'] != config_hash(cohort) or new_cohort['design'] != crossfit_design:
                raise ValueError('Row-crossfit documents do not follow the registered original cohort')
            config['row_crossfit'] = {'design':crossfit_design,'cohort_sha256':sha256_file(dest)}
            if row_refresh_sources is not None:
                from cellquant.row_refresh import COHORT_FILE as refresh_file, REFERENCE_FILE, DESIGN as refresh_design, read_crossfit_archive
                for name,path in zip((REFERENCE_FILE,refresh_file),row_refresh_sources,strict=True):
                    shutil.copyfile(path,refs/name)
                refresh = read_json(refs/refresh_file)
                parent_run,_ = read_crossfit_archive(refs/REFERENCE_FILE,refs/GRID_FILE,small=False)
                if (refresh['design'] != refresh_design or refresh['previous_cohort_hash'] != config_hash(new_cohort)
                        or refresh['crossfit_sha256'] != sha256_file(refs/REFERENCE_FILE) or not parent_run['crossfit_summary']['interaction_replicated']):
                    raise ValueError('Row refresh requires the completed reciprocal row experiment and registered cohort')
                config['row_refresh'] = {'design':refresh_design,'cohort_sha256':sha256_file(refs/refresh_file),
                                         'crossfit_sha256':refresh['crossfit_sha256']}
    if retest_pilot is not None:
        from cellquant.batch_retest import DESIGN as retest_design,read_document_pilot,retest_partitions
        from cellquant.gates import resolve_rules_path
        promotion,_,excluded=read_document_pilot(retest_pilot)
        if (promotion['next_step']!='formal_g1_retest'
                or promotion['fit_source']['parent_config_sha256']!=parent['parent_config_sha256']
                or promotion['fit_source']['parent_token_manifests']!=parent['parent_token_manifests']):
            raise ValueError('Document pilot does not qualify with these original inputs')
        config['batch_g1_retest']={**parent,'design':retest_design,'pilot_run_id':promotion['run_id'],
            'pilot_sha256':sha256_file(retest_pilot),'gate_rules_sha256':sha256_file(resolve_rules_path()),
            'excluded_sha256':sha256_file(atomic_json(output/'prior_documents.json',excluded))}
    if batch_screen is not None:
        from cellquant.batch_validation import DESIGN as batch_design, draw_fresh_documents, freeze_batch_inputs
        from cellquant.matching_batches import load_matching_arms
        if matching_archive is None:
            raise ValueError('Expected the matching archive used by the fit screen')
        config['batch_document_pilot']={**parent, **freeze_batch_inputs(
            output,Path(batch_screen),Path(matching_archive),config['seeds'],parent['parent_config_sha256'])}
    data=config['data'];model_id=config.get('model_id',config.get('formal_pilot',{}).get('model_id'))
    token=os.environ.get('HF_TOKEN') or None
    cache,dataset_cache=huggingface_cache_paths(config.get('runtime'))
    from transformers import AutoTokenizer
    from datasets import load_dataset
    if reuse_matching_inputs:
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
    if reuse_matching_inputs:
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
    if reuse_matching_inputs and cached_matching_model(config,model_id,weight_files,cache):
        config['runtime']['local_cache_only']=True
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=config['tokenizer_revision'],cache_dir=str(cache),token=token,
                                           local_files_only=reuse_matching_inputs and config['runtime'].get('local_cache_only',False),
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
    fresh=None
    if bilateral_sources is not None:
        if ds._fingerprint != cohort['dataset_fingerprint']:
            raise ValueError('The fresh cohort belongs to a different WikiText revision')
        fresh = draw_documents(blocks, screen, cohort['previous_42m_train_blocks'])
        if {str(s):[block_identity(b) for b in dev] for s, (_, dev) in fresh.items()} != cohort['heldout']:
            raise ValueError('Pinned 7B tokenizer or fresh article selection differs from the frozen list')
        if row_crossfit_cohort is not None:
            excluded = [*cohort['previous_42m_train_blocks'], *(b for docs in cohort['heldout'].values() for b in docs)]
            fresh = draw_documents(blocks,screen,excluded,design=crossfit_design)
            if {str(s):[block_identity(b) for b in dev] for s, (_,dev) in fresh.items()} != new_cohort['heldout']:
                raise ValueError('Pinned tokenizer or sampled row-crossfit documents differ from the frozen list')
            if row_refresh_sources is not None:
                excluded.extend(b for docs in new_cohort['heldout'].values() for b in docs)
                excluded.extend(refresh['previous_small_heldout'])
                fresh = draw_documents(blocks,screen,excluded,design=refresh_design)
                if {str(s):[block_identity(b) for b in dev] for s,(_,dev) in fresh.items()} != refresh['heldout']:
                    raise ValueError('Pinned tokenizer or sampled row-refresh documents differ from the frozen list')
        formal['heldout_sequences'] = bilateral_design['documents_per_seed']
        formal['heldout_tokens_per_panel'] = bilateral_design['documents_per_seed'] * bilateral_design['tokens_per_document']
    elif retest_pilot is not None:
        fresh=retest_partitions(blocks,previous,excluded)
        formal['heldout_sequences']=retest_design['dev']
        formal['heldout_tokens_per_panel']=retest_design['dev']*32
    elif previous is not None:
        fresh=confirmation_partitions(blocks,previous,formal['heldout_sequences'])
    if batch_screen is not None:
        reserved = [b for _, dev in fresh.values() for b in dev]
        arms = load_matching_arms(output/'fit_reference/measurements.zip',config['seeds'])
        fresh, exclusions = draw_fresh_documents(blocks, previous, reserved, arms)
        config['batch_document_pilot']['prior_documents_sha256']=sha256_file(atomic_json(output/'prior_documents.json',exclusions))
        formal['heldout_sequences'] = batch_design['documents_per_seed']
        formal['heldout_tokens_per_panel'] = batch_design['documents_per_seed']*batch_design['tokens_per_document']
    elif matching_study:
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
    if bilateral_sources is not None:
        protocol.update(bilateral_document_pilot=config['bilateral_document_pilot'],formal_pilot=formal)
        if row_crossfit_cohort is not None:
            protocol['row_crossfit'] = config['row_crossfit']
            if row_refresh_sources is not None:
                protocol['row_refresh'] = config['row_refresh']
    elif retest_pilot is not None:
        protocol.update(batch_g1_retest=config['batch_g1_retest'],formal_pilot=formal)
    elif batch_screen is not None:
        protocol.update(batch_document_pilot=config['batch_document_pilot'],formal_pilot=formal)
    elif matching_study:
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
    p.add_argument('--batch-document-screen',type=Path,help='Use the frozen rotations from this completed fit screen')
    p.add_argument('--matching-archive',type=Path,help='01d archive referenced by the fit screen')
    p.add_argument('--retest-pilot',type=Path,help='Verified document-pilot bundle for an independent formal batch G1 retest')
    p.add_argument('--bilateral-sources', type=Path, nargs=3, metavar=('FIT_ZIP','GRID_ZIP','COHORT'),
                   help='Freeze two completed rotations and prepare new articles for held-out scoring')
    p.add_argument('--row-crossfit-cohort',type=Path,help='Prepare a separate previously unused evaluation cohort for the row-swap experiment')
    p.add_argument('--row-refresh-sources',type=Path,nargs=2,metavar=('CROSSFIT_ZIP','COHORT'),help='Completed matched grids and unused articles for midpoint row switching')
    a=p.parse_args();prepare(a.config,a.output,include_tests=a.include_tests,relative_screen=a.relative_g1_screen,
                            matching_study=a.matching_study,batch_screen=a.batch_document_screen,matching_archive=a.matching_archive,
                            retest_pilot=a.retest_pilot,bilateral_sources=a.bilateral_sources,row_crossfit_cohort=a.row_crossfit_cohort,
                            row_refresh_sources=a.row_refresh_sources)
