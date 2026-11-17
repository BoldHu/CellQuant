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


def test_prepared_gpu_pilot_outputs_are_fixtures_and_not_gate_evidence(tmp_path,monkeypatch):
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
    rows=pilot.run_pilot(config,None,tmp_path/'pilot',graph_identity='fixture',execution_fingerprint='fixture',prepared=prepared)
    assert len(rows)==5
    for path in rows:
        row=read_json(path)
        assert row['source_type']=='unit_test_fixture' and row['status']=='COMPLETE'
        assert row['solver']['numerical_device']=='cuda:0'
        assert len(row['heldout'])==2
        assert Path(row['q_path']).is_file()
    with pytest.raises(ValueError,match='Only real-layer'):
        pilot.build_g1_evidence(rows,tmp_path/'rejected',g0_evidence_path='unused')
