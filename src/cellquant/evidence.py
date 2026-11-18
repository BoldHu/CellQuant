"""Build measured G2/G3 evidence. Unknown fields are omitted, never default PASS."""
from pathlib import Path
import math
import numpy as np
from .runtime import read_json,atomic_json,sha256_file
from .gates import evaluate_file
from .analysis import verified_results,make_paired_cell


def _write(stage,output,metrics,refs,provenance,source='real_model'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    evidence={'schema_version':'1.0','gate_id':stage,'source_type':source if refs else 'not_run','metrics':metrics,
              'artifact_refs':refs,'provenance':provenance if refs else {}}
    path=atomic_json(output/f'{stage}_evidence.json',evidence)
    report=evaluate_file(stage,path);atomic_json(output/f'{stage}.json',report);return report


def build_g2(paths,output,*,g1_evidence):
    rows,rejected=verified_results(paths)
    rows=[r for r in rows if r.get('development_only_evaluation') and r['method'] in {'cellquant','hadamard_shared','spinquant_full'}]
    refs=[{'path':r['source_path'],'sha256':r['source_sha256']} for r in rows];metrics={}
    by={(r['seed'],r['method']):r for r in rows}
    if len(by)!=len(rows):raise ValueError('Multiple canary cohorts; select one model and protocol')
    seeds=sorted({r['seed'] for r in rows})
    if rows:
        metrics={'g1_pass':evaluate_file('G1',g1_evidence)['status']=='PASS',
            'calibration_seed_count':len(seeds),'calibration_sequences':min(r['calibration_sequences'] for r in rows),
            'calibration_sequence_length':min(r['calibration_sequence_length'] for r in rows),
            'development_only_evaluation':all(r['development_only_evaluation'] for r in rows),
            'canonical_model_id':rows[0]['model_id'] if len({r['model_id'] for r in rows})==1 else 'MIXED',
            'quantization_setting':'W4A4KV4' if all(all(r['quantization'][k]==4 for k in ['w_bits','a_bits','k_bits','v_bits']) for r in rows) else 'MIXED',
            'rmsnorm_embedding_centering_disabled':all(r['rmsnorm_embedding_centering_disabled'] for r in rows),
            'gptq_pipeline_completed':all('dev' in r['raw'] for r in rows)}
        eq=[r['fp_equivalence'] for r in rows]
        if all(e and e.get('norm_fusion') for e in eq):
            legs=[leg for e in eq for leg in (e,e['norm_fusion'])]
            metrics.update(full_precision_model_equivalence_checked=True,
                fp32_logits_relative_rms_error=max(leg['relative_rms'] for leg in legs),
                fp32_mean_nll_abs_difference=max(abs(leg['mean_nll_delta']) for leg in legs))
        import torch
        schema_valid=True
        for r in rows:
            path=r['rotation_checkpoint_path']
            if sha256_file(path)!=r['rotation_checkpoint_sha256']:raise ValueError('Rotation checkpoint checksum changed')
            state=torch.load(path,map_location='cpu',weights_only=True)
            keys=[f'model.layers.{i}.self_attn.R2' for i in range(len(state)-1)]
            schema_valid &= set(state)=={'R1',*keys} and len(keys)>0
        metrics['rotation_checkpoint_schema_valid']=bool(schema_valid)
        complete=all((s,m) in by for s in seeds for m in ['cellquant','hadamard_shared','spinquant_full'])
        if complete:
            for s in seeds:
                if len({by[s,m]['protocol_sha256'] for m in ['cellquant','hadamard_shared','spinquant_full']})!=1:raise ValueError('Unmatched canary protocol')
            stock=all('our FP32' not in by[s,'spinquant_full']['implementation_variant'] for s in seeds)
            training=[by[s,'spinquant_full'].get('rotation_training') or {} for s in seeds]
            full=all(t.get('train_r2') is True and t.get('steps')==100 and t.get('processed_microbatches')==800
                     and t.get('data_stream_policy')=='all_train_documents_excluding_dev' for t in training)
            if stock and full:metrics['matched_spinquant_reproduction_present']=True
            pairs_h=[(by[s,'hadamard_shared'],by[s,'cellquant']) for s in seeds]
            pairs_s=[(by[s,'spinquant_full'],by[s,'cellquant']) for s in seeds]
            h=make_paired_cell(pairs_h,'dev');sp=make_paired_cell(pairs_s,'dev')
            dh=np.array(h['new_means'])-h['base_means'];ds=np.array(sp['new_means'])-sp['base_means']
            metrics.update(dev_ppl_gain_vs_hadamard_pct=float(100*(1-np.exp(dh.mean()))),
                max_seed_dev_ppl_regression_vs_spinquant_pct=float(np.max(100*np.expm1(ds))),
                total_calibration_speedup_vs_spinquant=float(min(b['total_calibration_seconds']/n['total_calibration_seconds'] for b,n in pairs_s)))
            if all('rtn_dev' in b['raw'] and 'rtn_dev' in n['raw'] for b,n in pairs_h):
                rt=make_paired_cell(pairs_h,'rtn_dev');dr=np.array(rt['new_means'])-rt['base_means']
                metrics['gptq_transfer_observed']=bool(np.all(dr<0) and np.all(dh<0))
                atomic_json(Path(output)/'rtn_gptq_transfer.json',{'seeds':seeds,'rtn_nll_delta':dr.tolist(),'gptq_nll_delta':dh.tolist()})
    return _write('G2',output,metrics,refs,{'rejected':rejected,'aggregation':'paired per-seed document NLL, maximum seed regression'})


def build_g3(output,*,g2_evidence,plan_path,hardware_probe_path,canary_paths,resume_audit_path,project_root):
    from .runner import execution_fingerprint,frozen_execution_config_hash
    plan=read_json(plan_path);probe=read_json(hardware_probe_path);resume=read_json(resume_audit_path)
    current=execution_fingerprint(project_root,plan['execution_provenance_path'])
    rows,rejected=verified_results(canary_paths)
    refs=[{'path':str(p),'sha256':sha256_file(p)} for p in [g2_evidence,plan_path,hardware_probe_path,resume_audit_path]]
    metrics={'g2_pass':evaluate_file('G2',g2_evidence)['status']=='PASS','artifact_hashes_verified':current['status']=='VERIFIED',
        'complete_run_manifest_written':bool(plan.get('jobs')) and plan.get('execution_fingerprint',{}).get('sha256')==current['sha256']}
    env=probe.get('environment',{});cuda=env.get('cuda',{});devices=cuda.get('devices',[])
    if devices:
        metrics['target_gpu_identity_verified']='RTX PRO 6000' in devices[0]['name'] and 'Blackwell' in devices[0]['name']
        metrics['visible_gpu_memory_gib']=devices[0]['total_gib']
    if probe.get('gemm_relative_error') is not None:metrics['cuda_gemm_smoke_pass']=probe['gemm_relative_error']<.02
    if probe.get('fht',{}).get('status') in {'PASS','FAIL'}:metrics['fht_backend_numeric_pass']=probe['fht']['status']=='PASS'
    if rows:metrics['calibration_peak_to_visible_memory_ratio']=max(r['peak_gpu_allocated_bytes']/r['visible_gpu_bytes'] for r in rows)
    # A CPU fixture or a different execution environment cannot qualify resume.
    if resume.get('source_type')=='real_model' and resume.get('execution_fingerprint_sha256')==current['sha256']:
        first=read_json(resume['uninterrupted_result_path']);second=read_json(resume['resumed_result_path'])
        if sha256_file(resume['uninterrupted_result_path'])!=resume['uninterrupted_sha256'] or sha256_file(resume['resumed_result_path'])!=resume['resumed_sha256']:
            raise ValueError('Resume result integrity mismatch')
        metrics['interrupt_resume_equivalence_pass']=(first['status']==second['status']=='COMPLETE' and
            second.get('parent_run_dir') is not None and first['metrics']==second['metrics'])
    protocol=current.get('payload',{}).get('provenance') or {}
    path=protocol.get('protocol_manifest_path')
    if path and Path(path).exists():metrics['protocol_frozen']=read_json(path).get('status')=='FROZEN'
    budget=plan.get('measured_budget_basis')
    if budget and rows:
        actual={r['run_id'] for r in rows}
        metrics['budget_plan_based_on_measurements']=set(budget.get('run_ids',[]))<=actual and bool(budget.get('run_ids'))
    access=[]
    for path in sorted((Path(project_root)/'runs'/'experiments').glob('*/formal_test_access.json')):
        access.append({'path':str(path),'sha256':sha256_file(path),'record':read_json(path)})
    access.sort(key=lambda r:r['record']['created_at_utc'])
    first_protocol={};changes=0
    for item in access:
        record=item['record'];model=record['model_id'];protocol_hash=record['protocol_sha256']
        if model in first_protocol and first_protocol[model]!=protocol_hash:changes+=1
        first_protocol.setdefault(model,protocol_hash)
    audit=atomic_json(Path(output)/'formal_test_access_audit.json',{'entries':access,'protocol_changes_after_first_test':changes,
        'scope':'All formal-scoring entries generated by this backend in this project run tree'})
    refs.append({'path':str(audit),'sha256':sha256_file(audit)})
    metrics['final_test_tuning_count']=changes
    configs=[]
    for job in plan.get('jobs',[]):
        path=Path(job['config_path'])
        if not path.is_file() or sha256_file(path)!=job['config_sha256']:raise ValueError('G3 plan config changed or missing')
        configs.append(frozen_execution_config_hash(read_json(path)))
    return _write('G3',output,metrics,refs,{'execution_fingerprint_sha256':current['sha256'],'rejected':rejected,
        'plan_path':str(Path(plan_path).resolve()),'plan_sha256':sha256_file(plan_path),'authorized_config_sha256':configs})


def write_resume_audit(original,resumed,output):
    rows,rejected=verified_results([original,resumed])
    if rejected or len(rows)!=2:raise ValueError('Two distinct completed real-model runs are required')
    a,b=rows
    fields=['model_id','model_revision','method','seed','protocol_sha256','dtype','execution_fingerprint_sha256']
    if any(a[k]!=b[k] for k in fields) or not b.get('parent_run_dir'):raise ValueError('Unmatched resume cohort or missing parent run')
    return atomic_json(output,{'source_type':'real_model','execution_fingerprint_sha256':a['execution_fingerprint_sha256'],
        'uninterrupted_result_path':str(Path(original).resolve()),'uninterrupted_sha256':sha256_file(original),
        'resumed_result_path':str(Path(resumed).resolve()),'resumed_sha256':sha256_file(resumed),
        'metrics_exactly_equal':a['metrics']==b['metrics'],'scope':'Observed completed pipeline outputs; target G3 is recomputed separately'})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='stage',required=True)
    g2=sub.add_parser('G2');g2.add_argument('--runs',required=True);g2.add_argument('--g1-evidence',required=True);g2.add_argument('--output',required=True)
    g3=sub.add_parser('G3')
    for name in ['output','g2-evidence','plan','hardware-probe','canary-runs','resume-audit']:g3.add_argument('--'+name,required=True)
    g3.add_argument('--project-root',default='.')
    audit=sub.add_parser('resume-audit')
    for name in ['original','resumed','output']:audit.add_argument('--'+name,required=True)
    a=p.parse_args()
    if a.stage=='G2':print(build_g2(sorted(Path(a.runs).rglob('result.json')),a.output,g1_evidence=a.g1_evidence)['status'])
    elif a.stage=='G3':print(build_g3(a.output,g2_evidence=a.g2_evidence,plan_path=a.plan,hardware_probe_path=a.hardware_probe,
        canary_paths=sorted(Path(a.canary_runs).rglob('result.json')),resume_audit_path=a.resume_audit,project_root=a.project_root)['status'])
    else:print(write_resume_audit(a.original,a.resumed,a.output))
