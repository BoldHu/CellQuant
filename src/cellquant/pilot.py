"""Real-layer shared-Q pilot and G1 aggregation from measured artifacts only."""
from __future__ import annotations
from dataclasses import asdict
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


def panel_metrics(panels,q,specs,device='cpu'):
    rows=[]
    for panel in panels:
        state=PanelState(panel,q,*panel_specs(specs,panel),device=device)
        pieces=panel.metadata['tokens_per_block'];offset=0;blocks=[]
        for index,count in enumerate(pieces):
            error=state.error[:,offset:offset+count];target=state.target[:,offset:offset+count]
            blocks.append({'block_id':panel.metadata['block_ids'][index],'document_id':panel.metadata['document_ids'][index],
                'squared_error':float(np.square(error).sum()),'target_energy':float(np.square(target).sum()),'sampled_tokens':count})
            offset+=count
        rows.append({'panel_id':panel.panel_id,'placement':panel.placement,'layer':panel.metadata['layer'],
            'nmse':state.loss,'budget_w':state.bw,'budget_a':state.ba,'blocks':blocks,
            'weight_shape':list(panel.weight.shape),'activation_shape':list(panel.activation.shape),
            'weight_crest':float(np.max(np.abs(state.a))/max(np.sqrt(np.mean(state.a**2)),1e-30)),
            'activation_crest':float(np.max(np.abs(state.x))/max(np.sqrt(np.mean(state.x**2)),1e-30))})
    return rows


def run_pilot(config,model,output,*,graph_identity,device='cuda',execution_fingerprint=None):
    output=Path(output);formal=config['formal_pilot'];records=[]
    if set(formal['methods'])!=set(METHODS):raise ValueError('G1 requires Hadamard, cell, STE, grid, fixed-scale controls')
    quant=config['quantization']
    env=environment_report();hardware=config_hash({k:env[k] for k in ['python','packages','cuda','platform']})
    for seed in config['seeds']:
        manifest_path=config['data']['token_manifests'][str(seed)]
        manifest,parts=load_token_manifest(manifest_path);fit=parts['fit'];heldout=parts['dev'];verify_disjoint(fit,heldout)
        start=time.perf_counter()
        kwargs=dict(layers=formal['layers'],projections=formal['projections'],input_weight_rows=formal['input_weight_rows'],seed=seed,device=device,graph_identity=graph_identity)
        fit_panels=capture_panels(model,fit,tokens_per_panel=formal['fit_tokens_per_panel'],**kwargs)
        from .spinquant_adapter import panel_quant_specs
        specs=panel_quant_specs(model,fit_panels,quant)
        capture_seconds=time.perf_counter()-start
        heldout_panels=capture_panels(model,heldout,tokens_per_panel=formal['heldout_tokens_per_panel'],**kwargs)
        q0=hadamard_rotation(model.config.hidden_size,seed)
        schedule=pair_schedule(len(q0),config['solver']['pair_budget'],seed)
        method_budget=config['solver'].get('wall_seconds_per_method',min(2400.,config.get('gpu_hour_budget',4)*3600/len(METHODS)/len(config['seeds'])))
        for method in formal['methods']:
            dest=output/f'seed_{seed}'/method;dest.mkdir(parents=True,exist_ok=True)
            resume=False
            if config.get('resume_run_dir'):
                import shutil
                prior=Path(config['resume_run_dir'])/f'seed_{seed}'/method
                saved=read_json(prior/'pilot_result.json') if (prior/'pilot_result.json').exists() else None
                if saved and saved['status']=='COMPLETE':
                    if sha256_file(saved['q_path'])!=saved['q_sha256'] or sha256_file(saved['data_manifest_path'])!=saved['data_manifest_sha256']:
                        raise ValueError('Completed pilot checkpoint integrity mismatch')
                    shutil.copy2(saved['q_path'],dest/'Q.npz');saved['q_path']=str((dest/'Q.npz').resolve())
                    saved['resumed_from']=str(prior.resolve());records.append(atomic_json(dest/'pilot_result.json',saved));continue
                if (prior/'rotation_state.npz').exists() and method in {'cellquant','fixed_scale_search_matched'}:
                    shutil.copy2(prior/'rotation_state.npz',dest/'rotation_state.npz');resume=True
                elif saved and method in {'givens_ste_matched','angle_grid_matched'}:
                    raise ValueError('Partial grid/STE resume is unsupported; do not reset its wall budget silently')
            cfg=SolverConfig(theta_limit=config['solver'].get('theta_limit',np.pi/4),
                event_limit_per_pair=config['solver'].get('event_limit_per_pair',200000),
                seconds_per_pair=config['solver'].get('seconds_per_pair',300.),wall_seconds=method_budget,
                device=config['solver'].get('device','cpu'),checkpoint_path=str(dest/'rotation_state.npz'),
                execution_fingerprint=execution_fingerprint or graph_identity,resume=resume)
            result=fit_baseline(method,fit_panels,q0,specs,schedule,cfg)
            q=result.pop('q');atomic_npz(dest/'Q.npz',Q=q)
            row={'source_type':'real_layer','status':result['status'],'seed':seed,'method':method,
                'model_id':formal['model_id'],'model_revision':config['model_revision'],
                'model_parameter_billions':sum(p.numel() for p in model.parameters())/1e9,
                'graph_identity':graph_identity,'hardware_sha256':hardware,'quantization':quant,
                'schedule_sha256':config_hash(schedule),'wall_budget_seconds':method_budget,
                'data_manifest_path':str(Path(manifest_path).resolve()),'data_manifest_sha256':sha256_file(manifest_path),
                'fit_panel_fingerprint':panel_fingerprint(fit_panels),'fit_documents':sorted({b['document_id'] for b in fit}),
                'heldout_documents':sorted({b['document_id'] for b in heldout}),
                'context_lengths':sorted({len(b['tokens']) for b in fit+heldout}),
                'capture_seconds':capture_seconds,'total_calibration_seconds':capture_seconds+result['seconds'],
                'q_path':str((dest/'Q.npz').resolve()),'q_sha256':sha256_file(dest/'Q.npz'),'solver':result,
                'heldout':panel_metrics(heldout_panels,q,specs,cfg.device)}
            path=atomic_json(dest/'pilot_result.json',row);records.append(path)
    return records


def build_g1_evidence(paths,output,*,g0_evidence_path):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not paths:
        evidence={'schema_version':'1.0','gate_id':'G1','source_type':'not_run','metrics':{},'provenance':{},'artifact_refs':[]}
    else:
        rows=[];refs=[]
        for path in paths:
            row=read_json(path)
            if row.get('source_type')!='real_layer':raise ValueError('Only real-layer artifacts can form G1')
            for k,h in [('q_path','q_sha256'),('data_manifest_path','data_manifest_sha256')]:
                if sha256_file(row[k])!=row[h]:raise ValueError('Pilot artifact checksum mismatch')
            rows.append(row);refs.append({'path':str(Path(path).resolve()),'sha256':sha256_file(path)})
        seeds=sorted({r['seed'] for r in rows});by={(r['seed'],r['method']):r for r in rows}
        if len(by)!=len(rows):raise ValueError('Duplicate seed/method pilot results')
        complete=all((s,m) in by and by[s,m]['status']=='COMPLETE' for s in seeds for m in METHODS)
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
