"""Configuration-driven real experiments, with fail-closed scientific gates.

CLI: python -m cellquant.experiment --config PATH [--preflight]
Missing revisions, data, protocols, gates or resources produce a BLOCKED record,
never a synthetic completed experiment. Each invocation has a unique run_id.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import contextlib
import copy
import gc
import hashlib
import json
import math
import os
import time
import uuid
from zipfile import BadZipFile
import numpy as np
from .runtime import atomic_json,read_json,sha256_file,config_hash,environment_report,utc_now,huggingface_cache_paths
from .data import immutable_revision,load_token_manifest,verify_disjoint
from .gates import evaluate_file
from .events import ResourceLimit


BACKEND_CONTRACT_VERSION='1.0'


class ExperimentBlocked(RuntimeError):pass


def _model_solver_config(settings, output, fingerprint):
    """Use the backend recorded in the frozen model-level protocol."""
    from .solver import SolverConfig
    backend = settings.get('backend', 'numpy_reference')
    tensor = backend == 'torch'
    return SolverConfig(
        backend=backend, device=settings.get('device', 'cuda:0' if tensor else 'cpu'),
        compute_dtype=settings.get('compute_dtype', 'float64'), cell_batch_size=settings.get('cell_batch_size', 1024),
        theta_limit=settings.get('theta_limit', math.pi/4), wall_seconds=settings.get('wall_seconds', 3600),
        fit_objective=settings.get('fit_objective', 'sum_nmse'),
        event_limit_per_pair=settings.get('event_limit_per_pair', 200000), seconds_per_pair=settings.get('seconds_per_pair', 300),
        checkpoint_path=str(output / ('rotation_state.pt' if tensor else 'rotation_state.npz')),
        execution_fingerprint=fingerprint)


def preflight(config,root=None):
    root=Path(root or Path.cwd()).resolve();missing=[];env=environment_report();gpu_memory=None
    stage=config.get('stage');data=config.get('data',{})
    if stage not in {'G1','G2','canary','full'}:missing.append('Unknown experiment stage')
    quant=config.get('quantization',{})
    if any(quant.get(k)!=4 for k in ['w_bits','a_bits','k_bits','v_bits']):missing.append('This main protocol requires W4A4KV4')
    if any(quant.get(k) is not True for k in ['symmetric_w','symmetric_a','symmetric_k','symmetric_v']):
        missing.append('Asymmetric scale event geometry is not implemented by the main solver')
    if any(quant.get(k,1.)!=1. for k in ['a_clip_ratio','k_clip_ratio','v_clip_ratio']):missing.append('Main solver requires fixed no-clipping activation/KV protocol')
    if config.get('result_path'):
        result_path=root/config['result_path']
        if result_path.exists() and read_json(result_path).get('status')=='COMPLETE':missing.append('Completed job result exists; resume through runner or use a new job id')
    for key in ['model_revision','tokenizer_revision','dataset_revision']:
        if not immutable_revision(config.get(key)):missing.append(f'Missing immutable {key}; run scripts/prepare_experiment.py')
    source_manifest=config.get('model_source_manifest_path')
    if not source_manifest or not Path(source_manifest).is_file():missing.append('Pinned model shard size/SHA-256 metadata missing')
    elif sha256_file(source_manifest)!=config.get('model_source_manifest_sha256'):missing.append('Model source metadata checksum mismatch')
    if stage=='G1':
        if (config.get('solver',{}).get('fit_objective')=='relative_site_nmse'
                and not config.get('g1_confirmation')):
            missing.append('Prepare the relative-objective confirmation with fresh held-out documents')
        manifests=data.get('token_manifests',{})
        if config.get('g1_confirmation'):
            protocol=Path(config.get('protocol_manifest_path') or '.')
            if not protocol.is_file() or sha256_file(protocol)!=config.get('protocol_sha256'):
                missing.append('Missing or changed confirmation protocol')
            else:
                frozen=read_json(protocol)
                frozen_manifests=frozen.get('token_manifest_sha256',{})
                if (frozen.get('g1_confirmation')!=config['g1_confirmation'] or frozen.get('solver')!=config.get('solver')
                        or set(manifests)!=set(frozen_manifests) or
                        any(not Path(p).is_file() or sha256_file(p)!=frozen_manifests.get(s) for s,p in manifests.items())):
                    missing.append('Confirmation objective or fresh validation manifests changed')
        for seed in config.get('seeds',[]):
            if str(seed) not in manifests or not Path(manifests[str(seed)]).is_file():missing.append(f'Missing real fit/dev token manifest for seed {seed}')
        g0=config.get('g0_evidence_path','runs/build_m2/G0_evidence.json')
        if not Path(g0).is_file() or evaluate_file('G0',g0)['status']!='PASS':missing.append('G0 measured correctness evidence missing/not passing')
    else:
        path=data.get('token_manifest_path')
        if not path or not Path(path).is_file():missing.append('Missing real token manifest')
        required='G1' if stage in {'G2','canary'} else 'G3'
        gate_evidence=config.get('gate_evidence_path',f'runs/gates/{required}_evidence.json')
        if not Path(gate_evidence).is_file() or evaluate_file(required,gate_evidence)['status']!='PASS':missing.append(f'Recomputed {required} evidence is required')
        if required=='G1' and Path(gate_evidence).is_file():
            provenance=read_json(gate_evidence).get('provenance',{})
            if provenance.get('pilot_variant')=='output_matching_batches':
                missing.append('This canary fits the legacy solver. A batch-rotation canary adapter is required for this G1 evidence.')
            modes=provenance.get('fit_objectives',['sum_nmse'])
            if (modes!=[config.get('solver',{}).get('fit_objective','sum_nmse')]
                    or (modes==['relative_site_nmse'] and not provenance.get('includes_original_grid'))):
                missing.append('G2 requires the fitted G1 objective and its original-grid comparison')
        protocol=config.get('protocol_manifest_path')
        if not protocol or not Path(protocol).is_file():missing.append('Missing frozen protocol manifest')
        elif sha256_file(protocol)!=config.get('protocol_sha256'):missing.append('Protocol checksum mismatch')
        elif read_json(protocol).get('status')!='FROZEN':missing.append('Protocol is not frozen')
        if stage=='full':
            from .runner import execution_fingerprint,frozen_execution_config_hash
            fingerprint=execution_fingerprint(root,config.get('execution_provenance_path'))
            missing.extend(fingerprint['missing'])
            code_git=fingerprint['payload']['code_git']
            if not code_git['commit'] or not code_git['tracked_clean'] or code_git['untracked_source_files']:
                missing.append('Formal runs require a committed clean code checkout; a source archive hash alone is not a code commit')
            if Path(gate_evidence).is_file() and read_json(gate_evidence).get('provenance',{}).get('execution_fingerprint_sha256')!=fingerprint['sha256']:
                missing.append('G3 is not bound to current execution fingerprint')
            if Path(gate_evidence).is_file() and frozen_execution_config_hash(config) not in read_json(gate_evidence).get('provenance',{}).get('authorized_config_sha256',[]):
                missing.append('This exact frozen job config is not included in the measured G3 plan')
            if not immutable_revision(config.get('harness_commit')):missing.append('Pinned harness commit required for formal tasks')
            if not config.get('evaluation',{}).get('task_metric_keys'):missing.append('Frozen six-task metric keys required')
            from .evaluation import TASK_METRICS,HARNESS_COMMIT
            if config.get('harness_commit')!=HARNESS_COMMIT:missing.append('Harness commit has not been audited by this adapter')
            task_revs=config.get('evaluation',{}).get('task_dataset_revisions',{})
            if set(task_revs)!=set(TASK_METRICS) or not all(immutable_revision(v) for v in task_revs.values()):
                missing.append('Six immutable task dataset revisions required')
            if not config.get('evaluation',{}).get('formal_test_access',False):missing.append('Formal test access is locked')
    if not env['cuda'].get('available'):missing.append('CUDA unavailable for model stages')
    elif stage!='G1':
        memory=env['cuda']['devices'][0]['total_gib']
        if memory<24 and '7b' in config.get('model_id','').lower():missing.append(f'Visible GPU memory {memory:.2f} GiB is below this full 7B implementation working set')
    import psutil
    runtime=config.get('runtime',{});resident=runtime.get('model_device','cpu').startswith('cuda')
    if stage=='G1' and resident:
        if runtime.get('cpu_offload',True):missing.append('GPU-resident G1 requires cpu_offload=false')
        if config.get('solver',{}).get('backend')!='torch' or not config.get('solver',{}).get('device','').startswith('cuda'):
            missing.append('GPU-resident G1 requires the torch CUDA solver, not the NumPy reference')
        if env['cuda'].get('available') and source_manifest and Path(source_manifest).is_file():
            size=sum(f['size_bytes'] for f in read_json(source_manifest)['files'])/2**30
            estimate=size*(2 if runtime.get('dtype')=='float32' else 1)+4
            import torch
            device=torch.device(runtime['model_device'])
            free,total=torch.cuda.mem_get_info(device)
            free_gib,total_gib=free/2**30,total/2**30
            fraction=runtime.get('max_memory_fraction',.85)
            budget_gib=total_gib*fraction
            name=torch.cuda.get_device_name(device)
            limits=[]
            if total_gib<estimate:limits.append('total_capacity')
            if budget_gib<estimate:limits.append('configured_budget')
            if free_gib<estimate:limits.append('currently_free')
            gpu_memory={'status':'BLOCKED' if limits else 'READY',
                'device':str(device),'name':name,'observed_at_utc':utc_now(),
                'total_gib':total_gib,'free_gib':free_gib,'budget_gib':budget_gib,
                'effective_available_gib':min(free_gib,budget_gib),'max_memory_fraction':fraction,
                'estimated_required_gib':estimate,'checkpoint_files_gib':size,'workspace_estimate_gib':4.,
                'insufficient_limits':limits,'current_process_pid':os.getpid(),
                'current_process_allocated_gib':torch.cuda.memory_allocated(device)/2**30,
                'current_process_reserved_gib':torch.cuda.memory_reserved(device)/2**30,
                'scope':'Engineering estimate and current device memory snapshot; not a measured peak or scientific gate.'}
            if limits:
                hint=('Selected GPU capacity is below the estimate. ' if 'total_capacity' in limits else '')
                if 'currently_free' in limits:
                    hint+='Inspect GPU processes with nvidia-smi; current-process allocator statistics do not include other kernels or CUDA libraries. '
                if 'configured_budget' in limits:
                    hint+='The configured memory budget is also below the estimate. '
                missing.append(f'GPU-resident capture estimates {estimate:.2f} GiB (checkpoint bytes + workspace); '
                    f'{name} [{device}]: total={total_gib:.2f}, free={free_gib:.2f}, '
                    f'budget={budget_gib:.2f} GiB (fraction={fraction}). '
                    f'Insufficient limits: {", ".join(limits)}. {hint}No CPU fallback.')
    model_id=config.get('model_id',config.get('formal_pilot',{}).get('model_id',''))
    if any(s in model_id.lower() for s in ['7b','8b']):
        available=psutil.virtual_memory().available/2**30
        if not resident and available<24:missing.append(f'Available CPU RAM {available:.2f} GiB below 24 GiB model/panel staging requirement')
    dtype=config.get('runtime',{}).get('dtype')
    if dtype not in {'float16','bfloat16','float32'}:missing.append('Record an explicit actual model dtype; no silent fallback')
    if dtype=='bfloat16' and env['cuda'].get('available') and not env['cuda'].get('bf16_supported'):missing.append('BF16 is unsupported on this GPU')
    overlay=config.get('spinquant_overlay','runs/spinquant_overlay_v2')
    if not (Path(overlay)/'cellquant_overlay.json').is_file():missing.append('Audited SpinQuant overlay missing')
    method=config.get('method')
    if method in {'spinquant_full','spinquant_r1_control'} and not config.get('spinquant_training'):
        missing.append('Explicit SpinQuant CE steps, accumulation and training token partition required')
    return {'status':'READY' if not missing else 'BLOCKED','missing':missing,'environment':env,'stage':stage,'gpu_memory':gpu_memory,
            'scope':'Execution prerequisites only; no new scientific evidence'}


def _load_model(config,*,training=False):
    import torch
    from transformers import AutoTokenizer,LlamaForCausalLM
    if training:
        from train_utils.modeling_llama_quant import LlamaForCausalLM as Model
    else:Model=LlamaForCausalLM
    model_id=config.get('model_id',config.get('formal_pilot',{}).get('model_id'))
    token=os.environ.get('HF_TOKEN') or None
    cache=str(huggingface_cache_paths(config.get('runtime'))[0])
    resident=config.get('runtime',{}).get('model_device','cpu').startswith('cuda')
    placement={'device_map':{'':config['runtime']['model_device']}} if resident else {}
    model=Model.from_pretrained(model_id,revision=config['model_revision'],token=token,cache_dir=cache,
        torch_dtype=getattr(torch,config['runtime']['dtype']),low_cpu_mem_usage=True,attn_implementation='eager',
        local_files_only=config.get('runtime',{}).get('local_cache_only',False),**placement)
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=config['tokenizer_revision'],token=token,cache_dir=cache,
        use_fast=True,add_bos_token=False,add_eos_token=False,local_files_only=config.get('runtime',{}).get('local_cache_only',False))
    model.eval();model.config.use_cache=False
    if not training:model.requires_grad_(False)
    if resident and any(p.device.type!='cuda' for p in model.parameters()):
        raise ExperimentBlocked('GPU-resident loading left CPU parameters; no automatic offload allowed')
    if getattr(model.config,'_commit_hash',None)!=config['model_revision']:raise ExperimentBlocked('Loaded model revision differs from frozen config')
    return model,tokenizer


def _verify_weight_files(config,output):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    model_id=config.get('model_id',config.get('formal_pilot',{}).get('model_id'))
    directory=Path(snapshot_download(model_id,revision=config['model_revision'],local_files_only=config.get('runtime',{}).get('local_cache_only',False),
        allow_patterns=['*.safetensors','*.safetensors.index.json'],token=os.environ.get('HF_TOKEN') or None,
        cache_dir=str(huggingface_cache_paths(config.get('runtime'))[0])))
    files=[]
    source=read_json(config['model_source_manifest_path'])
    if source['revision']!=config['model_revision']:raise ExperimentBlocked('Weight source revision mismatch')
    expected_source={r['path']:r for r in source['files']}
    weights=sorted(directory.glob('*.safetensors'))
    if not weights:raise ExperimentBlocked('No locally cached safetensors shards to verify')
    index=directory/'model.safetensors.index.json'
    if index.exists():
        expected=set(read_json(index)['weight_map'].values())
        if expected!={p.name for p in weights}:raise ExperimentBlocked('Missing/extra model shard relative to safetensors index')
    for path in weights:
        size=path.stat().st_size
        try:
            with safe_open(path,framework='pt',device='cpu') as f:
                for key in f.keys():f.get_slice(key).get_shape()
        except Exception as exc:
            raise ExperimentBlocked(f'Corrupt weight: path={path.resolve()}, size={size}, validation={type(exc).__name__}: {exc}') from exc
        digest=sha256_file(path);expected=expected_source.get(path.name)
        if expected is None or size!=expected['size_bytes'] or digest!=expected['sha256']:
            raise ExperimentBlocked(f'Weight checksum failed: path={path.resolve()}, size={size}, sha256={digest}, expected={expected}')
        files.append({'path':str(path.resolve()),'size_bytes':size,'sha256':digest,'hf_lfs_sha256_verified':True})
    atomic_json(Path(output)/'weights_manifest.json',{'revision':config['model_revision'],'files':files})


def run_experiment(config,*,project_root=None,preflight_only=False,profile_pair=False,profile_pairs=None,
                   diagnose_pairs=False,screen_objective=False,matching_study=False,matching_batch_archive=None,
                   batch_documents=False,batch_retest=False,all_controls=False,bilateral_archive=None,bilateral_documents=False,
                   bilateral_row_reference=None,row_crossfit=False):
    config=read_json(config) if isinstance(config,(str,Path)) else copy.deepcopy(config)
    profile_pair=profile_pair or profile_pairs is not None
    batch_screen = matching_batch_archive is not None
    bilateral_screen = bilateral_archive is not None
    row_diagnostic = bilateral_row_reference is not None
    if row_diagnostic and bilateral_documents:
        raise ValueError('Choose either row diagnosis or the original document pilot')
    bilateral_documents = bilateral_documents or row_diagnostic
    if all_controls and not batch_retest:raise ValueError('The complete-controls option belongs to the batch G1 retest')
    if sum((bool(profile_pair), diagnose_pairs, screen_objective, matching_study, batch_screen, batch_documents,
            batch_retest, bilateral_screen, bilateral_documents, row_crossfit)) > 1:
        raise ValueError('Choose either a single profiling mode or a single diagnostic per run')
    profile_count=1 if profile_pairs is None else profile_pairs
    if profile_pair:
        from .pilot import validate_profile_pair_count
        validate_profile_pair_count(config,profile_count)
    root=Path(project_root or Path.cwd()).resolve()
    run_id=utc_now().replace(':','').replace('+','_')+'_'+uuid.uuid4().hex[:10]
    output=root/'runs'/'experiments'/run_id;output.mkdir(parents=True)
    atomic_json(output/'config.json',config)
    # The diagnostic writes only its own run directory and optional invocation index.
    diagnostic = (diagnose_pairs or screen_objective or matching_study or batch_screen or batch_documents or
                  batch_retest or bilateral_screen or bilateral_documents or row_crossfit)
    checks = {k: v for k, v in config.items() if k != 'result_path'} if diagnostic else config
    ready=preflight(checks,root)
    if batch_screen or bilateral_screen:
        fit_archive = root / (matching_batch_archive if batch_screen else bilateral_archive)
        if not fit_archive.is_file():
            cohort = '01d' if batch_screen else 'Batch retest'
            ready['missing'].append(f'{cohort} measurement archive missing: {fit_archive}')
            ready['status']='BLOCKED'
        if config.get('solver', {}).get('fit_objective', 'sum_nmse') != 'sum_nmse':
            ready['missing'].append('Use the original sum-NMSE objective for the 01d fit replay')
            ready['status']='BLOCKED'
    if profile_pair and (config.get('stage')!='G1' or not config.get('runtime',{}).get('model_device','').startswith('cuda') or config.get('resume_run_dir')):
        ready['missing'].append('Pair profile requires a GPU-resident G1 config without resume_run_dir')
        ready['status']='BLOCKED'
    if diagnostic and (config.get('stage') != 'G1' or config.get('resume_run_dir')
            or not config.get('runtime', {}).get('model_device', '').startswith('cuda')
            or config.get('solver', {}).get('backend') != 'torch'
            or not config['solver'].get('device', '').startswith('cuda')
            or config['solver'].get('compute_dtype') != 'float64'):
        ready['missing'].append('Diagnostic requires a fresh GPU-resident G1 run with the Torch FP64 solver')
        ready['status']='BLOCKED'
    if bool(config.get('output_matching_study')) != matching_study:
        ready['missing'].append('Use the matching-study runner with its own frozen document cohort')
        ready['status']='BLOCKED'
    if bool(config.get('batch_g1_retest')) != (batch_retest or bilateral_screen):
        ready['missing'].append('Use the batch G1 runner or its fit-only angle screen with the independently prepared retest cohort')
        ready['status']='BLOCKED'
    elif batch_retest or bilateral_screen:
        from .batch_retest import verify_protocol
        try:
            verify_protocol(config)
            if bilateral_screen and fit_archive.is_file():
                from .bilateral_screen import retest_grid_rows
                retest_grid_rows(fit_archive, config)
        except (OSError,ValueError,KeyError,BadZipFile) as exc:
            ready['missing'].append(str(exc))
            ready['status']='BLOCKED'
    if bool(config.get('batch_document_pilot')) != batch_documents:
        ready['missing'].append('Use the frozen batch document protocol with --batch-documents')
        ready['status']='BLOCKED'
    elif batch_documents:
        from .batch_validation import verify_document_protocol
        try:
            verify_document_protocol(config)
        except (ValueError, KeyError, OSError) as exc:
            ready['missing'].append(str(exc))
            ready['status']='BLOCKED'
    if bool(config.get('bilateral_document_pilot')) != (bilateral_documents or row_crossfit) or bool(config.get('row_crossfit')) != row_crossfit:
        ready['missing'].append('Choose the evaluation or row-crossfit runner bound to the prepared document cohort')
        ready['status']='BLOCKED'
    elif bilateral_documents or row_crossfit:
        from .bilateral_documents import verify_document_inputs
        try:
            frozen_angles = verify_document_inputs(config,row_crossfit=row_crossfit)
            if row_diagnostic and not (root/bilateral_row_reference).is_file():
                raise ValueError(f'Missing frozen document archive: {bilateral_row_reference}')
        except (ValueError, KeyError, OSError, BadZipFile) as exc:
            ready['missing'].append(str(exc))
            ready['status']='BLOCKED'
    if screen_objective or matching_study:
        from .objective_screen import validate_screen_config, verify_parent_manifests
        try:
            validate_screen_config(config)
            verify_parent_manifests(config)
            if matching_study:
                from .matching_study import DESIGN
                frozen=read_json(config['protocol_manifest_path'])
                if (frozen.get('output_matching_study') != config.get('output_matching_study')
                        or config.get('output_matching_study',{}).get('design') != DESIGN
                        or any(frozen.get(k) != config.get(k) for k in ('solver','quantization','rotation'))):
                    raise ValueError('Matching study design or rotation protocol changed after preparation')
        except ValueError as exc:
            ready['missing'].append(str(exc))
            ready['status']='BLOCKED'
    atomic_json(output/'preflight.json',ready)
    base={'schema_version':'1.0','run_id':run_id,'created_at_utc':utc_now(),'run_dir':str(output),
          'config_sha256':config_hash(config),'source_type':'not_run','status':'BLOCKED',
          'execution_mode':'pair_profile_only' if profile_pair else 'experiment'}
    if profile_pair:base['profile_pair_count']=profile_count
    if diagnose_pairs:base['execution_mode']='fit_pair_diagnostic'
    if screen_objective:base['execution_mode']='fit_objective_screen'
    if matching_study:base['execution_mode']='output_matching_study'
    if batch_screen:base['execution_mode']='fit_matching_batch_screen'
    if bilateral_screen:base['execution_mode']='bilateral_angle_fit_screen'
    if bilateral_documents:base['execution_mode']='frozen_bilateral_document_pilot'
    if row_diagnostic:base['execution_mode']='bilateral_row_diagnostic'
    if row_crossfit:base['execution_mode']='matched_row_crossfit'
    if batch_documents:base['execution_mode']='frozen_batch_document_pilot'
    if batch_retest:base['execution_mode']='batch_g1_retest'
    if preflight_only or ready['status']!='READY':
        result={**base,'readiness':ready,'status':'PREFLIGHT_ONLY' if preflight_only else 'BLOCKED'}
        atomic_json(output/'result.json',result);return result
    import torch
    from .spinquant_adapter import (activate_overlay,fixed_r2,fuse_norms,rotate_model,export_spinquant_rotation,
        install_online_graph,configure_quantization,gptq_quantize,compare_logits,measure_rtn_transfer,panel_quant_specs)
    from .solver import hadamard_rotation,pair_schedule
    from .baselines import fit_baseline
    from .evaluation import evaluate_nll,evaluate_tasks
    from .capture import capture_panels
    from .runner import execution_fingerprint
    from .resources import ResourceMonitor
    from .progress import RunProgress
    print(f'[CellQuant] run_dir={output} pid={os.getpid()}',flush=True)
    progress=RunProgress(output)
    monitor=ResourceMonitor().start()
    model=None;timings={};phase='initialization';training=None;result=None
    @contextlib.contextmanager
    def measured(name):
        nonlocal phase
        phase=name;progress.step(name)
        torch.cuda.synchronize();start=time.perf_counter()
        try:yield
        finally:
            torch.cuda.synchronize();timings[name]=timings.get(name,0.)+time.perf_counter()-start
            atomic_json(output/'timings.json',timings)
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.reset_peak_memory_stats()
        os.environ['CELLQUANT_FHT_BACKEND']=config.get('runtime',{}).get('fht_backend','extension')
        overlay=activate_overlay(config.get('spinquant_overlay','runs/spinquant_overlay_v2'))
        atomic_json(output/'upstream_manifest.json',overlay)
        stage=config['stage'];method=config.get('method');seed=config.get('seed',config.get('seeds',[11])[0])
        with measured('load'):
            _verify_weight_files(config,output)
            model,tokenizer=_load_model(config,training=method in {'spinquant_full','spinquant_r1_control'})
        resident=config.get('runtime',{}).get('model_device','cpu').startswith('cuda')
        devices={}
        for p in model.parameters():devices[str(p.device)]=devices.get(str(p.device),0)+p.numel()*p.element_size()
        placement_report={'parameter_bytes_by_device':devices,'all_parameters_cuda':all(p.device.type=='cuda' for p in model.parameters()),'cpu_offload':not resident}
        atomic_json(output/'model_placement.json',placement_report)
        print('[CellQuant] model_placement='+json.dumps(placement_report),flush=True)
        progress.step('fixed_rotations')
        quant=config['quantization'];r2=fixed_r2(model.config,config['rotation']['fixed_r2_seed'],device=config['runtime'].get('model_device','cpu'))
        graph_identity=config_hash({'overlay':overlay['overlay_sha256'],'r2_seed':config['rotation']['fixed_r2_seed'],
                                     'model':config['model_revision'],'dtype':config['runtime']['dtype'],'r3r4':'upstream fp32_had',
                                     'r2_generator':'torch_cuda_hadamard_v1' if resident else 'numpy_pcg64_hadamard'})
        if stage=='G1':
            from .pilot import run_pilot,build_g1_evidence,summarize_pilot_results
            from .runner import frozen_execution_config_hash
            fingerprint=execution_fingerprint(root,config.get('execution_provenance_path'))
            if fingerprint['status']!='VERIFIED':raise ExperimentBlocked('G1 execution provenance incomplete')
            atomic_json(output/'execution_fingerprint.json',fingerprint)
            if config.get('resume_run_dir'):
                prior=Path(config['resume_run_dir'])
                if read_json(prior/'execution_fingerprint.json')['sha256']!=fingerprint['sha256'] or frozen_execution_config_hash(read_json(prior/'config.json'))!=frozen_execution_config_hash(config):
                    raise ExperimentBlocked('G1 resume provenance changed; create a new protocol')
                base['parent_run_dir']=str(prior.resolve())
            with measured('fusion'):
                progress.step('fusion.norms',device=str(next(model.parameters()).device));fuse_norms(model)
                progress.step('fusion.export',device=str(next(model.parameters()).device))
                identity=torch.eye(model.config.hidden_size,dtype=torch.float64,device=next(model.parameters()).device)
                export_spinquant_rotation(identity,r2,output/'fixed_R.bin');del identity
                progress.step('fusion.rotate',device='cuda');rotate_model(model,output/'fixed_R.bin')
                progress.step('fusion.online_graph');install_online_graph(model,quant,quantized=False)
                if resident:model.to(config['runtime']['model_device'])
            prepared=None
            if resident:
                from .pilot import prepare_pilot_inputs
                with measured('capture'):
                    prepared=prepare_pilot_inputs(config,model,graph_identity=graph_identity,
                                                  progress=progress.step,profile_only=profile_pair,fit_only=diagnose_pairs or batch_screen or bilateral_screen,
                                                  screen_objective=screen_objective,heldout_only=batch_documents or bilateral_documents)
                    if row_diagnostic or row_crossfit:
                        full_up_weight = model.model.layers[0].mlp.up_proj.module.weight.detach()
                del model;model=None;gc.collect();torch.cuda.empty_cache()
                captured_splits = [split for split in ('fit','screen_validation','heldout')
                                   if any(split in panels for panels in prepared['seeds'].values())]
                progress.step('g1.model_released',captured_seeds=list(prepared['seeds']),
                              captured_splits=captured_splits,
                              reason='requested panels captured and resident on CUDA')
            if batch_retest:
                from .batch_retest import run_retest
                summary=run_retest(config,prepared,output,graph_identity,all_controls=all_controls,progress=progress.step)
                result={**base,**summary,'timings':timings}
            elif row_crossfit:
                from .bilateral_documents import GRID_FILE
                from .row_crossfit import run_crossfit
                reference = Path(config['protocol_manifest_path']).parent/'fit_reference'/GRID_FILE
                summary = run_crossfit(config,prepared,full_up_weight,output,reference,fit_7b=frozen_angles,progress=progress.step)
                result = {**base,**summary,'protocol_sha256':config['protocol_sha256'],'timings':timings}
            elif bilateral_documents:
                from .bilateral_documents import FIT_FILE, GRID_FILE, document_decision
                from .bilateral_screen import score_frozen
                refs = Path(config['protocol_manifest_path']).parent/'fit_reference'
                summary = score_frozen(config,prepared,output,refs/GRID_FILE,refs/FIT_FILE,fit_7b=frozen_angles,progress=progress.step,
                                       document_reference=root/bilateral_row_reference if row_diagnostic else None,
                                       full_up_weight=full_up_weight if row_diagnostic else None)
                result = {**base,**summary,'fit_source':config['bilateral_document_pilot'],'protocol_sha256':config['protocol_sha256'],
                          'timings':timings}
                if not row_diagnostic:
                    decision = document_decision(summary['document_summary'])
                    result.update(pilot_decision=decision, next_step=decision['next_step'])
            elif batch_documents:
                from .batch_validation import run_document_pilot
                summary=run_document_pilot(config,prepared,output,progress=progress.step)
                result={**base,**summary,'timings':timings}
            elif batch_screen:
                from .matching_batches import run_batch_screen
                summary=run_batch_screen(config,prepared,output,fit_archive,progress=progress.step)
                result={**base,**summary,'timings':timings}
            elif bilateral_screen:
                from .bilateral_screen import run_screen
                summary=run_screen(config,prepared,output,fit_archive,progress=progress.step)
                result={**base,**summary,'timings':timings}
            elif matching_study:
                from .matching_study import run_matching_study
                summary=run_matching_study(config,prepared,output,execution_fingerprint=fingerprint['sha256'],progress=progress.step)
                result={**base,**summary,'timings':timings}
            elif screen_objective:
                from .objective_screen import run_objective_screen
                summary=run_objective_screen(config,prepared,output,execution_fingerprint=fingerprint['sha256'],progress=progress.step)
                result={**base,**summary,'timings':timings}
            elif diagnose_pairs:
                from .pair_diagnostic import run_pair_diagnostic
                summary=run_pair_diagnostic(config,prepared,output,progress=progress.step)
                result={**base,**summary,'timings':timings}
            elif profile_pair:
                from .pilot import profile_prepared_pair
                summary=profile_prepared_pair(config,prepared,output,execution_fingerprint=fingerprint['sha256'],progress=progress.step,pair_count=profile_count)
                result={**base,**summary,'timings':timings}
            else:
                paths=run_pilot(config,model,output,graph_identity=graph_identity,execution_fingerprint=fingerprint['sha256'],progress=progress.step,prepared=prepared)
                progress.step('g1.evidence')
                g0=config.get('g0_evidence_path','runs/build_m2/G0_evidence.json')
                primary=[p for p in paths if p.name=='pilot_result.json']
                gate=build_g1_evidence(primary,output,g0_evidence_path=g0)
                if config.get('g1_confirmation'):
                    gate=build_g1_evidence(paths,output/'including_original_grid',g0_evidence_path=g0,include_original_grid=True)
                result={**base,'source_type':'real_layer',**summarize_pilot_results(paths),
                        'evidence_kind':'REAL_LAYER_PILOT','gate':gate['status'],
                        'raw_results':[str(p) for p in paths],'timings':timings}
        else:
            data_manifest,parts=load_token_manifest(config['data']['token_manifest_path']);verify_disjoint(parts['fit'],parts['dev'])
            if len(parts['fit'])!=config['data']['calibration_samples']:raise ExperimentBlocked('Calibration sample count does not match protocol')
            if any(len(b['tokens'])!=config['data']['sequence_length'] for b in parts['fit']):raise ExperimentBlocked('Calibration context length mismatch')
            fingerprint=execution_fingerprint(root,config.get('execution_provenance_path'))
            atomic_json(output/'execution_fingerprint.json',fingerprint)
            if fingerprint['status']!='VERIFIED':raise ExperimentBlocked('Execution provenance incomplete: '+str(fingerprint['missing']))
            q0=hadamard_rotation(model.config.hidden_size,seed);checkpoint=output/'R.bin'
            solver_cfg = _model_solver_config(config.get('solver', {}), output, fingerprint['sha256'])
            solver_state = Path(solver_cfg.checkpoint_path)
            resume_dir=Path(config['resume_run_dir']).resolve() if config.get('resume_run_dir') else None
            resume_signature=config_hash({'config':{k:v for k,v in config.items() if k not in {'resume_run_dir','result_path'}},'execution':fingerprint['sha256']})
            reused=False;training_resume=None
            if resume_dir:
                import shutil
                old_config=read_json(resume_dir/'config.json');old_fp=read_json(resume_dir/'execution_fingerprint.json')
                old_signature=config_hash({'config':{k:v for k,v in old_config.items() if k not in {'resume_run_dir','result_path'}},'execution':old_fp['sha256']})
                if old_signature!=resume_signature:raise ExperimentBlocked('Resume config/source/environment/model/data mismatch; fork a fresh protocol')
                base['parent_run_dir']=str(resume_dir)
                old_timing=read_json(resume_dir/'timings.json') if (resume_dir/'timings.json').exists() else {}
                for key in ['capture','solve','fusion','gptq']:timings[key]=old_timing.get(key,0.)
                if (resume_dir/'rotation_checkpoint.json').is_file():
                    saved=read_json(resume_dir/'rotation_checkpoint.json')
                    if saved['signature']!=resume_signature or sha256_file(resume_dir/'R.bin')!=saved['sha256']:
                        raise ExperimentBlocked('Rotation checkpoint integrity mismatch')
                    shutil.copy2(resume_dir/'R.bin',checkpoint);reused=True
                elif method in {'spinquant_full','spinquant_r1_control'}:
                    candidates=[p for p in (resume_dir/'training').glob('checkpoint-*') if (p/'rotation_signature.json').is_file()]
                    if not candidates:raise ExperimentBlocked('No completed rotation training checkpoint exists yet')
                    training_resume=str(max(candidates,key=lambda p:int(p.name.split('-')[-1])))
                elif method in {'cellquant','fixed_scale_search_matched'} and (resume_dir/solver_state.name).is_file():
                    shutil.copy2(resume_dir/solver_state.name,solver_state);solver_cfg.resume=True
                else:raise ExperimentBlocked('No recoverable method checkpoint; grid/STE partial resume is not implemented')
                # Training model class must not be used for fused inference.
                if reused and method in {'spinquant_full','spinquant_r1_control'}:
                    train_report=resume_dir/'training/training_report.json'
                    training=read_json(train_report) if train_report.is_file() else read_json(resume_dir/'result.json').get('rotation_training')
                    if not training:raise ExperimentBlocked('Completed rotation has no verified training record')
                    del model;gc.collect();torch.cuda.empty_cache();model,tokenizer=_load_model(config)
            if method in {'spinquant_full','spinquant_r1_control'} and not reused:
                from .spinquant_train import train_rotations
                train=config['spinquant_training'];partition=train.get('partition','spinquant_train')
                if partition not in parts:raise ExperimentBlocked('Dedicated SpinQuant CE token stream missing')
                with measured('solve'):
                    learned,training=train_rotations(model,parts[partition],q0,r2,output/'training',quantization=quant,seed=seed,
                        steps=train['steps'],accumulation_steps=train['accumulation_steps'],learning_rate=train['learning_rate'],
                        dtype=config['runtime']['dtype'],train_r2=method=='spinquant_full',execution_fingerprint=fingerprint['sha256'],resume=training_resume)
                training['data_stream_policy']=train['stream']
                atomic_json(output/'training/training_report.json',training)
                import shutil
                shutil.copy2(learned,checkpoint);del model;gc.collect();torch.cuda.empty_cache()
                model,tokenizer=_load_model(config)
            audit_ids=torch.tensor(parts['dev'][0]['tokens'][None,:min(64,len(parts['dev'][0]['tokens']))],device='cuda')
            if method!='fp_reference':
                runtime_dtype = getattr(torch,config['runtime']['dtype'])
                def audit_logits():
                    with torch.inference_mode():
                        return model(audit_ids,use_cache=False).logits.float().cpu()
                with measured('fp_reference'):
                    model.to('cuda')
                    native_original = audit_logits()
                    model.float()
                    original = native_original if runtime_dtype == torch.float32 else audit_logits()
                with measured('fusion'):
                    fuse_norms(model)
                with measured('fp_equivalence'):
                    fused_error = compare_logits(original, audit_logits(), token_ids=audit_ids)
                    if fused_error['relative_rms']>1e-4 or abs(fused_error['mean_nll_delta'])>1e-4:
                        atomic_json(output/'fp_equivalence.json',{'norm_fusion':fused_error})
                        raise ExperimentBlocked('FP32 norm fusion equivalence failed')
                with measured('fusion'):
                    model.cpu()
                    if method not in {'spinquant_full','spinquant_r1_control'} and not reused:
                        export_spinquant_rotation(np.eye(len(q0)),r2,output/'fixed_R.bin')
                        rotate_model(model,output/'fixed_R.bin');install_online_graph(model,quant,quantized=False)
                if method not in {'spinquant_full','spinquant_r1_control'} and not reused:
                    if method=='hadamard_shared':q=q0
                    else:
                        panel_cfg=config.get('panels',{'layers':[0,15,31],'projections':['q_proj','up_proj','down_proj'],'tokens_per_panel':128,'input_weight_rows':64})
                        with measured('capture'):
                            panels=capture_panels(model,parts['fit'],seed=seed,device='cuda',graph_identity=graph_identity,**panel_cfg)
                        settings=config['solver'];schedule=pair_schedule(len(q0),settings['pair_budget'],settings.get('schedule_seed',seed))
                        with measured('solve'):
                            fitted=fit_baseline(method,panels,q0,panel_quant_specs(model,panels,quant),schedule,solver_cfg)
                            q=fitted.pop('q');atomic_json(output/'solver.json',fitted)
                            if fitted['status']=='RESOURCE_BLOCKED':raise ResourceLimit(fitted['reason'])
                            if fitted['status']!='COMPLETE':raise ExperimentBlocked('Rotation search incomplete: '+str(fitted['reason']))
                        del panels
                    export_spinquant_rotation(q,r2,checkpoint)
                atomic_json(output/'rotation_checkpoint.json',{'signature':resume_signature,'sha256':sha256_file(checkpoint)})
                with measured('fusion'):
                    rotate_model(model,checkpoint,global_only=method not in {'spinquant_full','spinquant_r1_control'} and not reused)
                    if method in {'spinquant_full','spinquant_r1_control'} or reused:install_online_graph(model,quant,quantized=False)
                with measured('fp_equivalence'):
                    model.float().to('cuda')
                    rotated = audit_logits()
                    equivalence = compare_logits(original, rotated, token_ids=audit_ids)
                    equivalence['norm_fusion'] = fused_error
                    model.to(dtype=runtime_dtype)
                    native_rotated = rotated if runtime_dtype == torch.float32 else audit_logits()
                    equivalence['runtime_dtype'] = config['runtime']['dtype']
                    equivalence['runtime_dtype_rotation'] = compare_logits(native_original, native_rotated, token_ids=audit_ids)
                    equivalence['audit_token_sha256']=hashlib.sha256(audit_ids.cpu().numpy().tobytes()).hexdigest()
                    equivalence['basis']='FP32 from runtime-loaded base weights'
                    atomic_json(output/'fp_equivalence.json',equivalence)
                    if equivalence['relative_rms']>1e-4 or abs(equivalence['mean_nll_delta'])>1e-4:raise ExperimentBlocked('FP32 equivalence failed; quantization not executed')
                model.cpu()
                rtn_dev=None
                if config.get('measure_rtn_transfer',stage in {'G2','canary'}):
                    with measured('rtn_transfer_diagnostic'):
                        rtn_dev=measure_rtn_transfer(model,parts['dev'],quant,output/'rtn_dev_nll.jsonl',
                            metadata={'run_id':run_id,'model_id':config['model_id'],'method':method,'calibration_seed':seed,'dataset_revision':config['dataset_revision'],'dataset':'dev'})
                with measured('gptq'):
                    gptq_dir=(resume_dir if reused else output)/'gptq_checkpoint'
                    gptq_signature=config_hash({'resume':resume_signature,'rotation_sha256':sha256_file(checkpoint)})
                    gptq_quantize(model,[torch.tensor(b['tokens'][None,:]) for b in parts['fit']],quant,
                                  checkpoint_dir=gptq_dir,signature=gptq_signature)
                    effective=configure_quantization(model,quant);atomic_json(output/'effective_quantization.json',effective)
            else:model.to(dtype=getattr(torch,config['runtime']['dtype']))
            meta={'run_id':run_id,'model_id':config['model_id'],'model_revision':config['model_revision'],
                  'tokenizer_revision':config['tokenizer_revision'],'method':method,'calibration_seed':seed,
                  'calibration_dataset_revision':config['dataset_revision'],'protocol_sha256':config['protocol_sha256']}
            metrics={};raw={}
            if method!='fp_reference' and rtn_dev is not None:raw['rtn_dev']=rtn_dev
            datasets=['dev'] if stage in {'G2','canary'} else ['wikitext2_test','c4_test']
            with measured('evaluation'):
                if stage=='full':
                    atomic_json(output/'formal_test_access.json',{'run_id':run_id,'model_id':config['model_id'],
                        'protocol_sha256':config['protocol_sha256'],'created_at_utc':utc_now(),'protocol_status':config['protocol_status'],
                        'config_sha256':config_hash(config),'purpose':'final_scoring'})
                for dataset in datasets:
                    if dataset not in parts:raise ExperimentBlocked(f'Frozen evaluation tokens missing: {dataset}')
                    revision=config['evaluation']['c4']['revision'] if dataset=='c4_test' else config['dataset_revision']
                    row=evaluate_nll(model,parts[dataset],output/(dataset+'_nll.jsonl'),metadata={**meta,'dataset':dataset,'dataset_revision':revision})
                    raw[dataset]=row;metrics[dataset+'_ppl']=row['ppl'];metrics[dataset+'_nll']=row['nll_per_token']
                if stage=='full':raw['tasks']=evaluate_tasks(model,tokenizer,output,harness_path=config['evaluation']['harness_path'],
                    harness_commit=config['harness_commit'],task_metric_keys=config['evaluation']['task_metric_keys'],
                    task_revisions=config['evaluation']['task_dataset_revisions'],metadata=meta)
            result={**base,'source_type':'real_model','evidence_kind':'REAL_MODEL_EXPERIMENT','status':'COMPLETE',
                'model_id':config['model_id'],'model_revision':config['model_revision'],'method':method,'seed':seed,
                'protocol_sha256':config['protocol_sha256'],'dtype':config['runtime']['dtype'],'metrics':metrics,'raw':raw,
                'timings':timings,'total_calibration_seconds':sum(timings.get(k,0) for k in ['capture','solve','fusion','gptq']),
                'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
                'visible_gpu_bytes':torch.cuda.get_device_properties(0).total_memory,'fp_equivalence':equivalence if method!='fp_reference' else None,
                'extra_online_r1_ops':0,'backend':'dequantized_float_simulation','implementation_variant':overlay['scope'],
                'executed_quantization_setting':'FP_'+config['runtime']['dtype'] if method=='fp_reference' else 'W4A4KV4',
                'base_model_parameter_count':sum(p.numel() for p in model.parameters()),
                'execution_fingerprint_sha256':fingerprint['sha256']}
            result['rotation_training']=training
            result.update(calibration_sequences=len(parts['fit']),calibration_sequence_length=config['data']['sequence_length'],
                          quantization=quant,rotation_checkpoint_path=str(checkpoint),
                          rotation_checkpoint_sha256=sha256_file(checkpoint) if checkpoint.exists() else None,
                          rmsnorm_embedding_centering_disabled=method!='fp_reference',development_only_evaluation=stage in {'G2','canary'},
                          model_weight_manifest_path=str(output/'weights_manifest.json'),model_weight_manifest_sha256=sha256_file(output/'weights_manifest.json'))
        if stage=='G1':
            result.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                          visible_gpu_bytes=torch.cuda.get_device_properties(0).total_memory)
        result['resources']=monitor.stop()
        atomic_json(output/'resources.json',result['resources'])
        atomic_json(output/'result.json',result)
        if config.get('result_path') and not diagnostic:
            target=root/config['result_path']
            if target.exists() and read_json(target).get('status')=='COMPLETE':raise ExperimentBlocked('Completed result index already exists; choose a new job id')
            atomic_json(target,result)
        return result
    except (KeyboardInterrupt,Exception) as exc:
        token=os.environ.get('HF_TOKEN');message=f'{type(exc).__name__}: {exc}'
        if token:message=message.replace(token,'[REDACTED]')
        result={**base,'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'BLOCKED' if isinstance(exc,ExperimentBlocked) else 'RESOURCE_BLOCKED' if isinstance(exc,(torch.cuda.OutOfMemoryError,ResourceLimit)) else 'FAILED',
                'phase':progress.state['phase'],'reason':message,'timings':timings}
        result['resources']=monitor.stop();atomic_json(output/'resources.json',result['resources'])
        atomic_json(output/'result.json',result)
        if isinstance(exc,KeyboardInterrupt):raise
        return result
    finally:
        progress.close(result['status'] if result is not None else 'INTERRUPTED_OR_FAILED')
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--result-index',help='New pointer file for this invocation, without changing its frozen config')
    parser.add_argument('--profile-pair',action='store_true',help='Engineering diagnosis: first G1 pair on full fit panels for both event methods; no G1 evidence')
    parser.add_argument('--profile-pairs',type=int,help='Engineering diagnosis of the first N scheduled pairs with continuous Q updates; original method budget; no G1 evidence')
    parser.add_argument('--diagnose-pairs',action='store_true',help='Compare output error coverage and matched single-pair gains on fit panels at Q0')
    parser.add_argument('--screen-objective',action='store_true',help='Compare absolute and initial-relative fit losses on a split of the eight calibration documents')
    parser.add_argument('--matching-study',action='store_true',help='Measure output matching on a fresh frozen cohort')
    parser.add_argument('--screen-matching-batches',type=Path,metavar='01D_ZIP',
                        help='Use the original G1 config and replay archived 01d rotations on fit documents only')
    parser.add_argument('--batch-documents',action='store_true',help='Score frozen batch selections on the new document cohort')
    parser.add_argument('--screen-bilateral-angles', type=Path, metavar='RETEST_ZIP', help='Score input-aware proposals on the stopped retest fit panels')
    parser.add_argument('--bilateral-documents',action='store_true',help='Score the frozen A100 angle screen on a fresh paired article cohort')
    parser.add_argument('--bilateral-rows',type=Path,metavar='DOCUMENT_ZIP',help='Replay the scored articles with the complete first up projection')
    parser.add_argument('--crossfit-rows',action='store_true',help='Refit the matched grid on a disjoint row panel and score both fits on fresh documents')
    parser.add_argument('--batch-retest',action='store_true',help='Refit and validate the frozen batch algorithm on new calibration and validation documents')
    parser.add_argument('--all-controls',action='store_true',help='Finish every matched solver even if the parent already rules out both G1 routes')
    args=parser.parse_args()
    if args.result_index and Path(args.result_index).exists():raise FileExistsError('Result index already exists; choose a unique path')
    result=run_experiment(args.config,preflight_only=args.preflight,profile_pair=args.profile_pair,profile_pairs=args.profile_pairs,
                          diagnose_pairs=args.diagnose_pairs,screen_objective=args.screen_objective,matching_study=args.matching_study,
                          matching_batch_archive=args.screen_matching_batches,batch_documents=args.batch_documents,
                          batch_retest=args.batch_retest,all_controls=args.all_controls,bilateral_archive=args.screen_bilateral_angles,
                          bilateral_documents=args.bilateral_documents,bilateral_row_reference=args.bilateral_rows,row_crossfit=args.crossfit_rows)
    if args.result_index:atomic_json(args.result_index,{k:result[k] for k in ['run_id','run_dir','status']})
    display={'status':result['status'],'run_dir':result['run_dir'],'reason':result.get('reason'),
                      'incomplete_methods':result.get('incomplete_methods',[]),
                      'missing':result.get('readiness',{}).get('missing',[])}
    if 'profile_pair_count' in result:display['profile_pair_count']=result['profile_pair_count']
    if 'budget_analysis' in result:display['budget_analysis']=result['budget_analysis']
    if 'paired_fit_summary' in result:display['paired_fit_summary']=result['paired_fit_summary']
    if 'objective_screen' in result:display['objective_screen']=result['objective_screen']
    if 'matching_summary' in result:display['comparisons']=result['matching_summary']['comparisons']
    if 'fit_screen' in result:display.update(fit_screen=result['fit_screen'],next_step=result['next_step'])
    if 'by_seed' in result and 'next_step' in result:display.update(by_seed=result['by_seed'],next_step=result['next_step'])
    if 'document_summary' in result:display.update(document_summary=result['document_summary'],next_step=result['next_step'])
    if 'route_bounds' in result:display.update(route_bounds=result['route_bounds'],gate=result.get('gate'),
                                              complete_comparison=result['complete_comparison'])
    print(json.dumps(display,ensure_ascii=False,indent=2))
    return 0 if result['status'] in {'COMPLETE','PREFLIGHT_ONLY','PROFILE_COMPLETE','DIAGNOSTIC_COMPLETE','SCREEN_COMPLETE','STUDY_COMPLETE','RETEST_STOPPED_BY_BOUND'} else 2


if __name__=='__main__':raise SystemExit(main())
