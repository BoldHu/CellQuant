"""Execution diagnostics from fixtures never grant G1 scientific qualification."""
from cellquant.pilot import summarize_pilot_results
from cellquant.runtime import atomic_json


def test_resource_reasons_survive_top_level_summary(tmp_path):
    paths=[]
    for seed in [11,23]:
        for method,status in [('angle_grid_matched','COMPLETE'),('cellquant','RESOURCE_BLOCKED')]:
            row={'source_type':'unit_test_fixture','seed':seed,'method':method,'status':status,
                 'solver':{'reason':'Pair wall-clock budget exceeded: 300s' if status!='COMPLETE' else None,
                           'completed_pairs':0 if status!='COMPLETE' else 128,'requested_pairs':128,
                           'resource_diagnostics':{'phase':'roots'} if status!='COMPLETE' else None}}
            paths.append(atomic_json(tmp_path/f'{seed}_{method}.json',row))
    result=summarize_pilot_results(paths)
    assert result['status']=='RESOURCE_BLOCKED' and len(result['incomplete_methods'])==2
    assert 'seed=11 cellquant' in result['reason'] and 'seed=23 cellquant' in result['reason']
    assert '300s' in result['reason']
    assert all(r['completed_pairs']==0 and r['resource_diagnostics']=={'phase':'roots'} for r in result['incomplete_methods'])
    assert 'gate' not in result


def test_no_results_or_missing_reason_are_not_silently_successful(tmp_path):
    assert summarize_pilot_results([])['status']=='RESOURCE_BLOCKED'
    path=atomic_json(tmp_path/'old.json',{'source_type':'unit_test_fixture','status':'RESOURCE_BLOCKED',
        'seed':11,'method':'cellquant','solver':{'log':[]}})
    result=summarize_pilot_results([path])
    assert result['reason'] and result['incomplete_methods'][0]['completed_pairs']==0


def test_complete_execution_summary_does_not_mean_gate_pass(tmp_path):
    path=atomic_json(tmp_path/'complete.json',{'source_type':'unit_test_fixture','status':'COMPLETE'})
    result=summarize_pilot_results([path])
    assert result=={'status':'COMPLETE','reason':None,'incomplete_methods':[]}


def test_old_budget_truncated_complete_is_ineligible_for_summary_and_gate(tmp_path):
    from cellquant.baselines import METHODS
    from cellquant.pilot import build_g1_evidence
    from cellquant.runtime import read_json,sha256_file
    artifact=atomic_json(tmp_path/'fixture_artifact.json',{'source_type':'unit_test_fixture'})
    refs={k:str(artifact) for k in ['q_path','data_manifest_path']}
    refs.update({k:sha256_file(artifact) for k in ['q_sha256','data_manifest_sha256']})
    # Construct malformed historical rows for the rejection branch only.
    paths=[]
    for method in METHODS:
        row={**refs,'source_type':'real_layer','status':'COMPLETE','seed':11,'method':method,
             'model_parameter_billions':7,'hardware_sha256':'fixture','fit_documents':['fit'],
             'heldout_documents':['dev'],'context_lengths':[2048],'heldout':[],
             'solver':{'shared_global_r1':True,'requested_pairs':128,
                       'completed_pairs':6 if method=='angle_grid_matched' else 128}}
        paths.append(atomic_json(tmp_path/f'{method}.json',row))
    summary=summarize_pilot_results(paths)
    assert summary['status']=='RESOURCE_BLOCKED' and '6/128' in summary['reason']
    output=tmp_path/'gate'
    gate=build_g1_evidence(paths,output,g0_evidence_path='examples/G0_evidence_NOT_RUN.json')
    assert gate['status']=='INSUFFICIENT_EVIDENCE'
    assert read_json(output/'G1_evidence.json')['metrics']=={}
