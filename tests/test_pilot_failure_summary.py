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
