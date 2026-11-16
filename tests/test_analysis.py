import numpy as np
import pytest
from cellquant.analysis import paired_cluster_bootstrap,analyze
from cellquant.runtime import atomic_json


def test_cluster_bootstrap_token_weights_and_shared_seeds():
    cell={'seeds':[11,23,37],'cluster_key':'wiki','ids':['doc1','doc2'],
          'delta_sums':np.array([[-1.,-90.],[-1.,-90.],[-1.,-90.]]),
          'counts':np.array([[1.,100.]]*3)}
    result=paired_cluster_bootstrap([cell],repetitions=1000)
    assert result['delta']==pytest.approx(-91/101)
    assert result['ci95_low']<=result['delta']<=result['ci95_high']
    assert result['ci95_high']<0


def test_empty_analysis_cannot_generate_paper_results(tmp_path):
    r=analyze([],tmp_path)
    assert r['status']=='INSUFFICIENT_EVIDENCE'
    assert not (tmp_path/'main_table.csv').exists()
    assert not list(tmp_path.glob('*.png'))


def test_fixture_cannot_enter_analysis(tmp_path):
    p=atomic_json(tmp_path/'fixture.json',{'source_type':'unit_test_fixture','status':'COMPLETE','evidence_kind':'REAL_MODEL_EXPERIMENT'})
    result=analyze([p],tmp_path/'out')
    assert result['accepted_runs']==0 and result['status']=='INSUFFICIENT_EVIDENCE'


def test_mismatched_seed_bootstrap_rejected():
    cell={'seeds':[11,23],'cluster_key':'wiki','ids':['a','b'],
          'delta_sums':np.zeros((2,2)),'counts':np.ones((2,2))}
    with pytest.raises(ValueError):paired_cluster_bootstrap([cell,{**cell,'seeds':[11,37]}],repetitions=100)
