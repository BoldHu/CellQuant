"""Real-layer shared-Q pilot and G1 aggregation from measured artifacts only."""
from __future__ import annotations
from dataclasses import asdict
import math
from pathlib import Path
import time
import numpy as np
from .runtime import atomic_json,read_json,sha256_file,config_hash,environment_report
from .solver import (SolverConfig,PanelState,hadamard_rotation,pair_schedule,atomic_npz,panel_fingerprint,panel_specs)
from .quantization import QuantSpec
from .baselines import METHODS,fit_baseline
from .capture import capture_panels
from .data import verify_disjoint,load_token_manifest
from .gates import evaluate_file
from .progress import report


def _incomplete_reason(row):
    solver = row.get('solver', {})
    if row.get('status') != 'COMPLETE' or solver.get('status', 'COMPLETE') != 'COMPLETE':
        return solver.get('reason') or 'Incomplete method; no solver reason was recorded'
    if row.get('method') in METHODS and row['method'] != 'hadamard_shared':
        requested = solver.get('requested_pairs')
        done = solver.get('completed_pairs', len(solver.get('log', [])))
        if requested is None or requested <= 0 or done != requested or solver.get('stopped_by_wall_budget'):
            return f'Incomplete coordinate schedule: {done}/{requested} pairs'
        if solver.get('seconds', 0) > row.get('wall_budget_seconds', math.inf):
            return 'Total solver wall-clock budget exhausted'
    return None


def summarize_pilot_results(paths):
    """Summarize execution failures, without assigning a scientific gate result."""
    paths=list(paths);incomplete=[]
    for path in paths:
        row=read_json(path)
        reason = _incomplete_reason(row)
        if reason is None:continue
        solver=row.get('solver',{})
        incomplete.append({'seed':row.get('seed'),'method':row.get('method'),
            'status':'RESOURCE_BLOCKED' if row.get('status')=='COMPLETE' else row.get('status'),'reason':reason,
            'completed_pairs':solver.get('completed_pairs',len(solver.get('log',[]))),
            'requested_pairs':solver.get('requested_pairs'),
            'resource_diagnostics':solver.get('resource_diagnostics'),
            'result_path':str(path)})
    reason='; '.join(f"seed={r['seed']} {r['method']}: {r['reason']}" for r in incomplete) or None
    if not paths:reason='No pilot method results were produced'
    return {'status':'COMPLETE' if paths and not incomplete else 'RESOURCE_BLOCKED',
            'reason':reason,'incomplete_methods':incomplete}


def panel_metrics(panels,q,specs,device='cpu',*,progress=None):
    tensor=hasattr(q,'detach')
    if tensor:
        from .tensor_solver import PanelState as TensorState
    rows=[]
    for panel in panels:
        report(progress,'heldout.panel',panel_id=panel.panel_id,device=device)
        state=TensorState(panel,q,*panel_specs(specs,panel)) if tensor else PanelState(panel,q,*panel_specs(specs,panel),device=device)
        pieces=panel.metadata['tokens_per_block'];offset=0;blocks=[]
        for index,count in enumerate(pieces):
            error=state.error[:,offset:offset+count];target=state.target[:,offset:offset+count]
            blocks.append({'block_id':panel.metadata['block_ids'][index],'document_id':panel.metadata['document_ids'][index],
                'squared_error':float(error.square().sum() if tensor else np.square(error).sum()),
                'target_energy':float(target.square().sum() if tensor else np.square(target).sum()),'sampled_tokens':count})
            offset+=count
        rows.append({'panel_id':panel.panel_id,'placement':panel.placement,'layer':panel.metadata['layer'],
            'nmse':float(state.loss),'budget_w':float(state.bw),'budget_a':float(state.ba),'blocks':blocks,
            'weight_shape':list(panel.weight.shape),'activation_shape':list(panel.activation.shape),
            'weight_crest':float(state.a.abs().max()/state.a.square().mean().sqrt().clamp_min(1e-30)) if tensor else float(np.max(np.abs(state.a))/max(np.sqrt(np.mean(state.a**2)),1e-30)),
            'activation_crest':float(state.x.abs().max()/state.x.square().mean().sqrt().clamp_min(1e-30)) if tensor else float(np.max(np.abs(state.x))/max(np.sqrt(np.mean(state.x**2)),1e-30))})
        del state
    return rows


def prepare_pilot_inputs(config,model,*,graph_identity,progress=None,profile_only=False,fit_only=False):
    """Capture the panels consumed by this run before releasing the model."""
    from .spinquant_adapter import panel_quant_specs
    from .tensor_solver import synchronize
    device=config['runtime']['model_device'];formal=config['formal_pilot'];seeds={}
    requested_seeds = config['seeds'][:1] if profile_only else config['seeds']
    for seed in requested_seeds:
        _,parts=load_token_manifest(config['data']['token_manifests'][str(seed)])
        verify_disjoint(parts['fit'],parts['dev'])
        kwargs=dict(layers=formal['layers'],projections=formal['projections'],input_weight_rows=formal['input_weight_rows'],
                    seed=seed,device=device,graph_identity=graph_identity,resident=True)
        synchronize(device);start=time.perf_counter()
        fit=capture_panels(model,parts['fit'],tokens_per_panel=formal['fit_tokens_per_panel'],
            progress=lambda phase,**d:report(progress,'g1.'+phase,seed=seed,split='fit',**d),**kwargs)
        specs=panel_quant_specs(model,fit,config['quantization']);synchronize(device)
        seconds=time.perf_counter()-start
        seeds[seed]={'fit':fit,'specs':specs,'capture_seconds':seconds}
        if not (profile_only or fit_only):
            seeds[seed]['heldout']=capture_panels(model,parts['dev'],tokens_per_panel=formal['heldout_tokens_per_panel'],
                progress=lambda phase,**d:report(progress,'g1.'+phase,seed=seed,split='heldout',**d),**kwargs)
    return {'seeds':seeds,'hidden_size':model.config.hidden_size,
            'model_parameter_billions':sum(p.numel() for p in model.parameters())/1e9}


def pilot_method_budget(config):
    return config['solver'].get('wall_seconds_per_method',
        min(2400.,config.get('gpu_hour_budget',4)*3600/len(METHODS)/len(config['seeds'])))


def validate_profile_pair_count(config,pair_count):
    configured=config['solver']['pair_budget']
    if isinstance(pair_count,bool) or not isinstance(pair_count,int) or not 1<=pair_count<=configured:
        raise ValueError(f'Profile pair count must be an integer in [1, {configured}]')
    return pair_count


def analyze_profile_budget(records,configured_pairs,method_budget):
    """Conditional throughput projection, never a gate or full-run qualification.

    Pair search times exclude initialization, accepted-Q refresh and checkpoint
    overhead. Their extrapolation can expose a budget concern but cannot prove
    that the whole method will fit, even when the estimated component fits.
    """
    if configured_pairs<1 or not math.isfinite(method_budget) or method_budget<=0:
        raise ValueError('Positive configured pair count and method budget required')
    rows=[]
    for record in records:
        solver=record.get('solver',{});log=solver.get('log',[])
        completed=solver.get('completed_pairs',len(log));requested=solver.get('requested_pairs')
        seconds=[entry.get('seconds') for entry in log]
        row={'method':record.get('method'),'profile_status':record.get('status'),
             'completed_pairs':completed,'requested_profile_pairs':requested,
             'observed_method_seconds':solver.get('seconds'),'mean_pair_search_seconds':None,
             'projected_full_search_seconds':None,'projected_method_seconds':None,
             'projected_method_headroom_seconds':None,'assessment':'INCOMPLETE_PROFILE'}
        if record.get('status')=='COMPLETE' and completed==requested==len(log) and completed>0:
            if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>0 for v in seconds):
                row['assessment']='MISSING_VALID_TIMINGS'
            else:
                mean=sum(seconds)/completed;projection=mean*configured_pairs
                row.update(mean_pair_search_seconds=mean,min_pair_search_seconds=min(seconds),
                    max_pair_search_seconds=max(seconds),projected_full_search_seconds=projection,
                    assessment='PROJECTED_SEARCH_EXCEEDS_METHOD_BUDGET' if projection>method_budget else 'SEARCH_COMPONENT_ESTIMATE_WITHIN_BUDGET')
                loop = solver.get('this_invocation_pair_loop_seconds')
                elapsed = solver.get('seconds')
                if (isinstance(loop, (float, int)) and isinstance(elapsed, (float, int))
                        and math.isfinite(elapsed) and 0 < loop <= elapsed):
                    estimate = elapsed + (configured_pairs - completed) * loop / completed
                    row.update(projected_method_seconds=estimate,
                               projected_method_headroom_seconds=method_budget-estimate)
        rows.append(row)
    return {'configured_pair_count':configured_pairs,'method_budget_seconds':method_budget,
            'budget_seconds_per_configured_pair_before_other_overhead':method_budget/configured_pairs,
            'methods':rows,'full_run_readiness':'NOT_ESTABLISHED_BY_PROFILE',
            'assumption':'Remaining pairs have the same mean search time as this prefix; this is neither a runtime guarantee nor a confidence interval.',
            'excluded_costs':'Search-only estimates exclude setup, Q updates and checkpoints. Method estimates charge measured setup/finalization once and extrapolate the continuous pair loop including updates and checkpoints. Both exclude capture and held-out scoring.',
            'scope':'Engineering resource diagnosis only; no held-out result or G1 qualification.'}


def profile_prepared_pair(config,prepared,output,*,execution_fingerprint,progress=None,pair_count=1):
    """Bounded real-panel diagnosis; never writes G1 evidence or evaluates held-out."""
    from .tensor_solver import hadamard_rotation as tensor_hadamard
    if config['solver'].get('backend')!='torch':raise ValueError('Pair profiling requires the torch solver')
    pair_count=validate_profile_pair_count(config,pair_count)
    output=Path(output);seed=config['seeds'][0];ready=prepared['seeds'][seed]
    panels,specs=ready['fit'],ready['specs']
    sources={p.metadata.get('source_type') for p in panels}
    if len(sources)!=1 or not sources.issubset({'real_layer','unit_test_fixture'}):
        raise ValueError('Explicit panel provenance required for pair profiling')
    settings=config['solver'];device=settings['device']
    q0=tensor_hadamard(prepared['hidden_size'],seed,device)
    schedule=pair_schedule(len(q0),settings['pair_budget'],seed)
    if not schedule:raise ValueError('A nonempty pair schedule is required')
    method_budget=pilot_method_budget(config)
    records=[];paths=[]
    for method in ['fixed_scale_search_matched','cellquant']:
        def emit(phase,**details):report(progress,'profile.'+phase,seed=seed,method=method,**details)
        dest=output/'pair_profile'/f'seed_{seed}'/method;dest.mkdir(parents=True,exist_ok=True)
        cfg=SolverConfig(theta_limit=settings.get('theta_limit',np.pi/4),backend='torch',device=device,
            cell_batch_size=settings.get('cell_batch_size',1024),
            compute_dtype=settings.get('compute_dtype','float64'),event_limit_per_pair=settings.get('event_limit_per_pair',200000),
            seconds_per_pair=settings.get('seconds_per_pair',300.),wall_seconds=method_budget,
            checkpoint_path=str(dest/'rotation_state.pt'),execution_fingerprint=execution_fingerprint)
        emit('solve.start',wall_budget_seconds=method_budget,seconds_per_pair=cfg.seconds_per_pair,profile_pair_count=pair_count)
        result=fit_baseline(method,panels,q0,specs,schedule[:pair_count],cfg,progress=emit)
        q=result.pop('q');atomic_npz(dest/'Q.npz',Q=q)
        row={'source_type':next(iter(sources)),'evidence_kind':'PAIR_PROFILE_ONLY',
             'seed':seed,'method':method,'status':result['status'],'solver':result,
             'configured_pair_count':len(schedule),'profiled_pair':list(schedule[0]),
             'profiled_pair_count':pair_count,'profiled_pairs':[list(p) for p in schedule[:pair_count]],
             'scope':'Scheduled prefix with continuous Q updates on unchanged full fit panels. No held-out scoring or G1 qualification.'}
        path=atomic_json(dest/'profile_result.json',row);paths.append(str(path));records.append(row)
        emit('method.saved',solver_status=result['status'],solver_reason=result.get('reason'),result_path=str(path))
    incomplete=[{'method':r['method'],'reason':r['solver'].get('reason'),
                 'resource_diagnostics':r['solver'].get('resource_diagnostics')} for r in records if r['status']!='COMPLETE']
    summary={'source_type':next(iter(sources)),'evidence_kind':'PAIR_PROFILE_ONLY',
        'status':'RESOURCE_BLOCKED' if incomplete else 'PROFILE_COMPLETE',
        'reason':'; '.join(f"{r['method']}: {r['reason']}" for r in incomplete) or None,
        'incomplete_methods':incomplete,'profile_results':paths,'seed':seed,'profile_pair_count':pair_count,
        'budget_analysis':analyze_profile_budget(records,len(schedule),method_budget),
        'scope':'Engineering diagnosis only; not a full rotation fit or G1 evidence.'}
    atomic_json(output/'pair_profile.json',summary)
    return summary


def run_pilot(config,model,output,*,graph_identity,device='cuda',execution_fingerprint=None,progress=None,prepared=None):
    output=Path(output);formal=config['formal_pilot'];records=[]
    if set(formal['methods'])!=set(METHODS):raise ValueError('G1 requires Hadamard, cell, STE, grid, fixed-scale controls')
    quant=config['quantization']
    env=environment_report();hardware=config_hash({k:env[k] for k in ['python','packages','cuda','platform']})
    for seed in config['seeds']:
        def emit(phase,**details):report(progress,'g1.'+phase,seed=seed,**details)
        emit('tokens')
        manifest_path=config['data']['token_manifests'][str(seed)]
        manifest,parts=load_token_manifest(manifest_path);fit=parts['fit'];heldout=parts['dev'];verify_disjoint(fit,heldout)
        start=time.perf_counter()
        kwargs=dict(layers=formal['layers'],projections=formal['projections'],input_weight_rows=formal['input_weight_rows'],seed=seed,device=device,graph_identity=graph_identity)
        if prepared is None:
            fit_panels=capture_panels(model,fit,tokens_per_panel=formal['fit_tokens_per_panel'],progress=lambda phase,**d:emit(phase,split='fit',**d),**kwargs)
            from .spinquant_adapter import panel_quant_specs
            specs=panel_quant_specs(model,fit_panels,quant)
            capture_seconds=time.perf_counter()-start
            heldout_panels=capture_panels(model,heldout,tokens_per_panel=formal['heldout_tokens_per_panel'],progress=lambda phase,**d:emit(phase,split='heldout',**d),**kwargs)
            hidden=model.config.hidden_size;parameters=sum(p.numel() for p in model.parameters())/1e9
        else:
            ready=prepared['seeds'].pop(seed)
            fit_panels,heldout_panels,specs,capture_seconds=ready['fit'],ready['heldout'],ready['specs'],ready['capture_seconds']
            hidden=prepared['hidden_size'];parameters=prepared['model_parameter_billions']
            del ready
        tensor=config['solver'].get('backend')=='torch'
        emit('initial_rotation',device=config['solver'].get('device','cpu'))
        if tensor:
            from .tensor_solver import hadamard_rotation as tensor_hadamard
            q0=tensor_hadamard(hidden,seed,config['solver']['device'])
        else:q0=hadamard_rotation(hidden,seed)
        fit_hash=panel_fingerprint(fit_panels)
        sources={p.metadata.get('source_type') for p in [*fit_panels,*heldout_panels]}
        if len(sources)!=1 or not sources.issubset({'real_layer','unit_test_fixture'}):
            raise ValueError('Pilot panels must share an explicit real-layer or test-fixture source')
        source_type=next(iter(sources))
        schedule=pair_schedule(len(q0),config['solver']['pair_budget'],seed)
        method_budget=pilot_method_budget(config)
        for method in formal['methods']:
            def method_progress(phase,**details):emit(phase,method=method,**details)
            method_progress('solve.start',device=config['solver'].get('device','cpu'),wall_budget_seconds=method_budget)
            dest=output/f'seed_{seed}'/method;dest.mkdir(parents=True,exist_ok=True)
            resume=False
            if config.get('resume_run_dir'):
                import shutil
                prior=Path(config['resume_run_dir'])/f'seed_{seed}'/method
                saved=read_json(prior/'pilot_result.json') if (prior/'pilot_result.json').exists() else None
                if saved and _incomplete_reason(saved) is None:
                    if sha256_file(saved['q_path'])!=saved['q_sha256'] or sha256_file(saved['data_manifest_path'])!=saved['data_manifest_sha256']:
                        raise ValueError('Completed pilot checkpoint integrity mismatch')
                    shutil.copy2(saved['q_path'],dest/'Q.npz');saved['q_path']=str((dest/'Q.npz').resolve())
                    saved['resumed_from']=str(prior.resolve());records.append(atomic_json(dest/'pilot_result.json',saved));continue
                state_name='rotation_state.pt' if tensor else 'rotation_state.npz'
                if (prior/state_name).exists() and method in {'cellquant','fixed_scale_search_matched'}:
                    shutil.copy2(prior/state_name,dest/state_name);resume=True
                elif saved and method in {'givens_ste_matched','angle_grid_matched'}:
                    raise ValueError('Partial grid/STE resume is unsupported; do not reset its wall budget silently')
            cfg=SolverConfig(theta_limit=config['solver'].get('theta_limit',np.pi/4),
                cell_batch_size=config['solver'].get('cell_batch_size',1024),
                backend=config['solver'].get('backend','numpy_reference'),compute_dtype=config['solver'].get('compute_dtype','float64'),
                event_limit_per_pair=config['solver'].get('event_limit_per_pair',200000),
                seconds_per_pair=config['solver'].get('seconds_per_pair',300.),wall_seconds=method_budget,
                device=config['solver'].get('device','cpu'),checkpoint_path=str(dest/('rotation_state.pt' if tensor else 'rotation_state.npz')),
                execution_fingerprint=execution_fingerprint or graph_identity,resume=resume)
            result=fit_baseline(method,fit_panels,q0,specs,schedule,cfg,progress=method_progress)
            method_progress('solve.complete',solver_status=result['status'],solver_reason=result.get('reason'),
                            resource_diagnostics=result.get('resource_diagnostics'))
            q=result.pop('q');atomic_npz(dest/'Q.npz',Q=q)
            row={'source_type':source_type,'status':result['status'],'seed':seed,'method':method,
                'model_id':formal['model_id'],'model_revision':config['model_revision'],
                'model_parameter_billions':parameters,
                'graph_identity':graph_identity,'hardware_sha256':hardware,'quantization':quant,
                'schedule_sha256':config_hash(schedule),'wall_budget_seconds':method_budget,
                'data_manifest_path':str(Path(manifest_path).resolve()),'data_manifest_sha256':sha256_file(manifest_path),
                'fit_panel_fingerprint':fit_hash,'fit_documents':sorted({b['document_id'] for b in fit}),
                'heldout_documents':sorted({b['document_id'] for b in heldout}),
                'context_lengths':sorted({len(b['tokens']) for b in fit+heldout}),
                'capture_seconds':capture_seconds,'total_calibration_seconds':capture_seconds+result['seconds'],
                'q_path':str((dest/'Q.npz').resolve()),'q_sha256':sha256_file(dest/'Q.npz'),'solver':result,
                'heldout':panel_metrics(heldout_panels,q,specs,cfg.device,progress=method_progress) if result['status']=='COMPLETE' else []}
            path=atomic_json(dest/'pilot_result.json',row);records.append(path)
            method_progress('method.saved',result_path=str(path),solver_status=result['status'],solver_reason=result.get('reason'))
        del fit_panels,heldout_panels,q0
    return records


def build_g1_evidence(paths,output,*,g0_evidence_path):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not paths:
        evidence={'schema_version':'1.0','gate_id':'G1','source_type':'not_run','metrics':{},'provenance':{},'artifact_refs':[]}
    else:
        rows=[];refs=[]
        for path in paths:
            row=read_json(path)
            if row.get('evidence_kind')=='PAIR_PROFILE_ONLY':raise ValueError('Pair profiles cannot form G1 evidence')
            if row.get('source_type')!='real_layer':raise ValueError('Only real-layer artifacts can form G1')
            for k,h in [('q_path','q_sha256'),('data_manifest_path','data_manifest_sha256')]:
                if sha256_file(row[k])!=row[h]:raise ValueError('Pilot artifact checksum mismatch')
            rows.append(row);refs.append({'path':str(Path(path).resolve()),'sha256':sha256_file(path)})
        seeds=sorted({r['seed'] for r in rows});by={(r['seed'],r['method']):r for r in rows}
        if len(by)!=len(rows):raise ValueError('Duplicate seed/method pilot results')
        complete=all((s,m) in by and _incomplete_reason(by[s,m]) is None for s in seeds for m in METHODS)
        metrics={'g0_pass':evaluate_file('G0',g0_evidence_path)['status']=='PASS',
            'calibration_seed_count':len(seeds),'real_model_parameter_billions':min(r['model_parameter_billions'] for r in rows),
            'matched_solver_controls_complete':complete,
            'same_hardware_timing':len({r['hardware_sha256'] for r in rows})==1,
            'fit_sequence_count':min(len(r['fit_documents']) for r in rows),
            'heldout_sequence_count':min(len(r['heldout_documents']) for r in rows),
            'calibration_heldout_disjoint':all(not(set(r['fit_documents'])&set(r['heldout_documents'])) for r in rows),
            'shared_global_r1':all(r['solver']['shared_global_r1'] for r in rows),
            'real_panel_provenance_verified':all(r['context_lengths']==[2048] for r in rows)}
        panel_sets=[{p['panel_id'] for p in r['heldout']} for r in rows]
        same_sites=all(s==panel_sets[0] for s in panel_sets)
        metrics['site_count']=len(panel_sets[0]) if same_sites else 0
        metrics['layer_count']=len({p['layer'] for p in rows[0]['heldout']}) if same_sites else 0
        metrics['full_quant_groups_preserved']=all(p['weight_shape'][1]==p['activation_shape'][0] for r in rows for p in r['heldout'])
        metrics['down_projection_output_side_correct']=all(p['placement']=='output' for r in rows for p in r['heldout'] if p['panel_id'].endswith('down_proj')) and any(p['panel_id'].endswith('down_proj') for p in rows[0]['heldout'])
        if complete and same_sites:
            sites=sorted(panel_sets[0]);values={key:np.array([next(p['nmse'] for p in row['heldout'] if p['panel_id']==site) for site in sites]) for key,row in by.items()}
            controls=['givens_ste_matched','angle_grid_matched','fixed_scale_search_matched']
            # Same-seed controls must use identical frozen panels/schedule/time/R2.
            matched=all(len({by[s,m][k] for m in METHODS})==1 for s in seeds for k in ['fit_panel_fingerprint','schedule_sha256','wall_budget_seconds','graph_identity'])
            metrics['matched_solver_controls_complete']=matched
            new=np.stack([values[s,'cellquant'] for s in seeds]);had=np.stack([values[s,'hadamard_shared'] for s in seeds])
            best=np.stack([np.minimum.reduce([values[s,m] for m in controls]) for s in seeds])
            if np.any(had<=0) or np.any(best<=0):raise ValueError('Zero comparator NMSE; percentage improvements are undefined')
            gain_h=(100*(1-new/had)).mean(0);gain_b=(100*(1-new/best)).mean(0)
            metrics.update(each_seed_pooled_heldout_improved=bool(np.all(new.mean(1)<had.mean(1))),
                median_nmse_gain_vs_hadamard_pct=float(np.median(gain_h)),median_nmse_gain_vs_best_matched_solver_pct=float(np.median(gain_b)),
                max_site_nmse_regression_vs_best_pct=float(np.max(-gain_b)),site_win_fraction_vs_best_matched_solver=float(np.mean(gain_b>0)),
                median_nmse_regression_vs_best_matched_solver_pct=float(np.median(-gain_b)),site_noninferior_fraction_at_2pct=float(np.mean(gain_b>=-2)),
                pilot_calibration_speedup=float(np.mean([min(by[s,m]['total_calibration_seconds'] for m in controls)/by[s,'cellquant']['total_calibration_seconds'] for s in seeds])))
            atomic_json(output/'G1_distributions.json',{'seeds':seeds,'sites':sites,'new_nmse':new.tolist(),'hadamard_nmse':had.tolist(),'best_control_nmse':best.tolist(),
                'site_gain_hadamard_pct':gain_h.tolist(),'site_gain_best_control_pct':gain_b.tolist(),'scope':'exploratory raw seed/site distributions; no token-IID inference'})
        # A resource-interrupted comparison is incomplete evidence, not a
        # negative measurement of method efficacy. Preserve raw partial results.
        if not complete:metrics={}
        evidence={'schema_version':'1.0','gate_id':'G1','source_type':'real_layer','metrics':metrics,
            'provenance':{'raw_results':len(rows),'complete_comparison':complete,'g0_evidence_sha256':sha256_file(g0_evidence_path),
                          'aggregation':'mean seed-relative improvement per site, median sites'},'artifact_refs':refs}
    atomic_json(output/'G1_evidence.json',evidence);gate=evaluate_file('G1',output/'G1_evidence.json');atomic_json(output/'G1.json',gate)
    return gate
