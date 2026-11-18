"""GPU placement and complete pilot I/O fixtures, explicitly excluded from G1."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch=pytest.importorskip('torch')
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')


def test_loader_requests_gpu_and_freezes_base_weights(monkeypatch):
    import transformers
    from cellquant.experiment import _load_model
    model=torch.nn.Linear(4,4).cuda();model.config=SimpleNamespace(_commit_hash='a'*40,use_cache=True)
    observed={}
    def load(*args,**kwargs):observed.update(kwargs);return model
    monkeypatch.setattr(transformers.LlamaForCausalLM,'from_pretrained',load)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:object())
    cfg={'model_id':'unit_test_fixture','model_revision':'a'*40,'tokenizer_revision':'a'*40,
         'runtime':{'dtype':'float16','model_device':'cuda:0','local_cache_only':True}}
    actual,_=_load_model(cfg)
    assert observed['device_map']=={'':'cuda:0'} and observed['low_cpu_mem_usage']
    assert all(p.is_cuda and not p.requires_grad for p in actual.parameters())


@pytest.mark.parametrize('confirm', [False, True])
def test_prepared_gpu_pilot_outputs_are_fixtures_and_not_gate_evidence(tmp_path,monkeypatch,confirm):
    from cellquant import pilot
    from cellquant.solver import CalibrationPanel
    from cellquant.quantization import QuantSpec
    from cellquant.baselines import METHODS
    from cellquant.runtime import read_json
    rng=np.random.default_rng(412)
    def panels(label):
        result=[]
        for name,placement,shape in [('q_proj','input',(2,4)),('down_proj','output',(4,8))]:
            metadata={'source_type':'unit_test_fixture','layer':0,'block_ids':[label],
                      'document_ids':[label],'tokens_per_block':[2]}
            result.append(CalibrationPanel('layer_0.'+name,placement,
                torch.tensor(rng.normal(size=shape),device='cuda'),torch.tensor(rng.normal(size=(shape[1],2)),device='cuda'),metadata=metadata))
        return result
    manifest=tmp_path/'manifest.json';manifest.write_text('{"source_type":"unit_test_fixture"}')
    parts={name:[{'document_id':name,'document_sha256':name,'block_id':name,'tokens':np.arange(8),
                  'token_ids_sha256':name}] for name in ['fit','dev']}
    monkeypatch.setattr(pilot,'load_token_manifest',lambda path:({'source_type':'unit_test_fixture'},parts))
    prepared={'hidden_size':4,'model_parameter_billions':1e-6,'seeds':{11:{'fit':panels('fit'),
              'heldout':panels('dev'),'specs':QuantSpec(group_size=4),'capture_seconds':0.}}}
    config={'seeds':[11],'formal_pilot':{'methods':list(METHODS),'model_id':'unit_test_fixture',
        'layers':[0],'projections':['q_proj','down_proj'],'input_weight_rows':2},
        'data':{'token_manifests':{'11':str(manifest)}},'quantization':{},'model_revision':'a'*40,
        'solver':{'backend':'torch','device':'cuda','compute_dtype':'float64','pair_budget':1,'theta_limit':.01,
                  'wall_seconds_per_method':120,'seconds_per_pair':60}}
    if confirm:
        config['g1_confirmation']={'fixture':True}
        config['solver']['fit_objective']='relative_site_nmse'
    rows=pilot.run_pilot(config,None,tmp_path/'pilot',graph_identity='fixture',execution_fingerprint='fixture',prepared=prepared)
    assert len(rows)==(6 if confirm else 5)
    for path in rows:
        row=read_json(path)
        assert row['source_type']=='unit_test_fixture' and row['status']=='COMPLETE'
        assert row['solver']['numerical_device']=='cuda:0'
        expected='relative_site_nmse' if confirm and row['method']!='original_objective_grid' else 'sum_nmse'
        assert row['solver']['fit_objective']['mode']==expected
        assert (path.name=='original_grid_result.json')==(row['method']=='original_objective_grid')
        assert len(row['heldout'])==2
        assert Path(row['q_path']).is_file()
    with pytest.raises(ValueError,match='Only real-layer'):
        pilot.build_g1_evidence(rows,tmp_path/'rejected',g0_evidence_path='unused')


@pytest.mark.parametrize('pair_count',[1,8])
def test_pair_profile_reuses_full_panels_but_cannot_form_g1(tmp_path,pair_count):
    from cellquant import pilot
    from cellquant.solver import CalibrationPanel,pair_schedule
    from cellquant.quantization import QuantSpec
    from cellquant.runtime import read_json,atomic_json
    rng=np.random.default_rng(718)
    panels=[CalibrationPanel('fixture.q','input',torch.tensor(rng.normal(size=(2,4)),device='cuda'),
        torch.tensor(rng.normal(size=(4,2)),device='cuda'),metadata={'source_type':'unit_test_fixture'}),
        CalibrationPanel('fixture.down','output',torch.tensor(rng.normal(size=(4,8)),device='cuda'),
        torch.tensor(rng.normal(size=(8,2)),device='cuda'),metadata={'source_type':'unit_test_fixture'})]
    prepared={'hidden_size':4,'seeds':{11:{'fit':panels,'specs':QuantSpec(group_size=4)}}}
    config={'seeds':[11,23],'solver':{'backend':'torch','device':'cuda','pair_budget':128,
        'theta_limit':.01,'seconds_per_pair':60,'wall_seconds_per_method':120}}
    before=json.dumps(config,sort_keys=True)
    result=pilot.profile_prepared_pair(config,prepared,tmp_path,execution_fingerprint='fixture',pair_count=pair_count)
    assert json.dumps(config,sort_keys=True)==before
    assert result['status']=='PROFILE_COMPLETE' and result['evidence_kind']=='PAIR_PROFILE_ONLY'
    assert result['source_type']=='unit_test_fixture' and len(result['profile_results'])==2
    assert result['profile_pair_count']==pair_count
    assert result['budget_analysis']['method_budget_seconds']==120
    assert result['budget_analysis']['full_run_readiness']=='NOT_ESTABLISHED_BY_PROFILE'
    assert not list(tmp_path.rglob('pilot_result.json')) and not (tmp_path/'G1_evidence.json').exists()
    for path in result['profile_results']:
        row=read_json(path)
        assert row['configured_pair_count']==128 and row['solver']['requested_pairs']==pair_count
        assert row['profiled_pairs']==[list(p) for p in pair_schedule(4,128,11)[:pair_count]]
        assert [r['pair'] for r in row['solver']['log']]==row['profiled_pairs']
        assert row['solver']['completed_pairs']==pair_count
        assert row['profiled_pair']==list(pair_schedule(4,128,11)[0])
        row['source_type']='real_layer'
        rejection=atomic_json(tmp_path/'rejected_profile.json',row)
        with pytest.raises(ValueError,match='Pair profiles cannot form G1'):
            pilot.build_g1_evidence([rejection],tmp_path/'no_gate',g0_evidence_path='unused')
