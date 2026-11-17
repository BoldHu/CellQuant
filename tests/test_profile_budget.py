"""Engineering extrapolation is explicitly separated from scientific evidence."""
import math
import sys
import pytest

from cellquant.pilot import analyze_profile_budget,pilot_method_budget


def row(seconds,status='COMPLETE'):
    return {'source_type':'unit_test_fixture','method':'cellquant','status':status,
            'solver':{'seconds':180.,'completed_pairs':len(seconds),'requested_pairs':len(seconds),
                      'log':[{'seconds':s} for s in seconds]}}


def test_actual_g1_budget_allocation_and_conditional_projection():
    config={'seeds':[11,23],'gpu_hour_budget':4,'solver':{'wall_seconds':3600}}
    assert pilot_method_budget(config)==1440.
    r=analyze_profile_budget([row([20.]*8)],128,pilot_method_budget(config))
    m=r['methods'][0]
    assert m['projected_full_search_seconds']==2560
    assert r['budget_seconds_per_configured_pair_before_other_overhead']==11.25
    assert m['assessment']=='PROJECTED_SEARCH_EXCEEDS_METHOD_BUDGET'
    assert r['full_run_readiness']=='NOT_ESTABLISHED_BY_PROFILE'
    assert 'gate' not in r
    config['solver']['wall_seconds_per_method']=1200
    assert pilot_method_budget(config)==1200


def test_fast_prefix_does_not_certify_full_run_or_gate():
    r=analyze_profile_budget([row([3.,4.,5.])],128,1440.)
    assert r['methods'][0]['assessment']=='SEARCH_COMPONENT_ESTIMATE_WITHIN_BUDGET'
    assert r['methods'][0]['mean_pair_search_seconds']==4.
    assert r['full_run_readiness']=='NOT_ESTABLISHED_BY_PROFILE'


@pytest.mark.parametrize('seconds',[[None],[math.nan],[math.inf],[0.],[-1.]])
def test_missing_or_invalid_timings_never_become_resource_success(seconds):
    r=analyze_profile_budget([row(seconds)],128,1440.)
    assert r['methods'][0]['assessment']=='MISSING_VALID_TIMINGS'
    assert r['methods'][0]['projected_full_search_seconds'] is None


def test_incomplete_profile_keeps_actual_completed_count():
    data=row([1.,2.],status='RESOURCE_BLOCKED');data['solver']['requested_pairs']=8
    r=analyze_profile_budget([data],128,1440.)
    assert r['methods'][0]['completed_pairs']==2
    assert r['methods'][0]['assessment']=='INCOMPLETE_PROFILE'


@pytest.mark.parametrize('count',[0,-1,129,True,1.5])
def test_invalid_prefix_is_rejected_before_preflight_or_model_loading(tmp_path,monkeypatch,count):
    from cellquant import experiment
    def forbidden(*args,**kwargs):raise AssertionError('Invalid prefix must fail before model/preflight work')
    monkeypatch.setattr(experiment,'preflight',forbidden)
    with pytest.raises(ValueError,match='Profile pair count'):
        experiment.run_experiment({'solver':{'pair_budget':128}},project_root=tmp_path,profile_pairs=count)
    assert not (tmp_path/'runs').exists()


def test_cli_prefix_selects_engineering_mode(monkeypatch,capsys):
    from cellquant import experiment
    calls=[]
    def run(config,**kwargs):
        calls.append((config,kwargs))
        return {'status':'PROFILE_COMPLETE','run_dir':'fixture','profile_pair_count':8,
                'budget_analysis':{'scope':'unit_test_fixture'}}
    monkeypatch.setattr(experiment,'run_experiment',run)
    monkeypatch.setattr(sys,'argv',['cellquant.experiment','--config','fixture.json','--profile-pairs','8'])
    assert experiment.main()==0
    assert calls[0][1]['profile_pairs']==8
    assert 'budget_analysis' in capsys.readouterr().out
