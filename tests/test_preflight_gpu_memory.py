"""Memory preflight fixtures; none of these records are scientific evidence."""
import json
from pathlib import Path

import pytest

torch=pytest.importorskip('torch')


@pytest.mark.parametrize('total,free,fraction,dtype,limits',[
    (22.0343017578125,14.13360595703125,.85,'bfloat16',['currently_free']),
    (22.0343017578125,21.,.85,'bfloat16',[]),
    (22.0343017578125,21.,.5,'bfloat16',['configured_budget']),
    (15.,14.,.85,'float16',['total_capacity','configured_budget','currently_free']),
    (22.0343017578125,21.,.85,'float32',['total_capacity','configured_budget','currently_free']),
])
def test_preflight_distinguishes_capacity_free_memory_and_budget(tmp_path,monkeypatch,total,free,fraction,dtype,limits):
    from cellquant import experiment
    from cellquant.runtime import atomic_json,sha256_file
    config=json.loads(Path('configs/colab_pilot.json').read_text(encoding='utf-8'))
    config.update({k:'a'*40 for k in ['model_revision','tokenizer_revision','dataset_revision']})
    source=atomic_json(tmp_path/'weights.json',{'source_type':'unit_test_fixture',
        'files':[{'size_bytes':int(12.55*2**30)}]})
    config.update(model_source_manifest_path=str(source),model_source_manifest_sha256=sha256_file(source))
    fixture=atomic_json(tmp_path/'fixture.json',{'source_type':'unit_test_fixture'})
    config['data']['token_manifests']={str(s):str(fixture) for s in config['seeds']}
    config['g0_evidence_path']=str(fixture)
    monkeypatch.setattr(experiment,'evaluate_file',lambda *a,**kw:{'status':'PASS'})
    overlay=tmp_path/'overlay';overlay.mkdir()
    atomic_json(overlay/'cellquant_overlay.json',{'source_type':'unit_test_fixture'})
    config['spinquant_overlay']=str(overlay)
    # The selected device must be queried, not whichever device is current.
    config['runtime'].update(model_device='cuda:1',dtype=dtype,max_memory_fraction=fraction)
    config['solver']['device']='cuda:1'
    monkeypatch.setattr(experiment,'environment_report',lambda:{'cuda':{
        'available':True,'bf16_supported':True,
        'devices':[{'index':0,'name':'fixture GPU 0','total_gib':4.},
                   {'index':1,'name':'fixture GPU 1','total_gib':total}]}})
    queried=[]
    def memory(device):
        queried.append(str(device))
        return int(free*2**30),int(total*2**30)
    monkeypatch.setattr(torch.cuda,'mem_get_info',memory)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda dev:'fixture GPU 1')
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda dev:0)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda dev:0)
    result=experiment.preflight(config,root=tmp_path)
    report=result['gpu_memory']
    assert queried==['cuda:1']
    assert result['status']==report['status']==('BLOCKED' if limits else 'READY')
    assert report['insufficient_limits']==limits
    assert report['free_gib']==pytest.approx(free)
    assert report['budget_gib']==pytest.approx(total*fraction)
    assert report['estimated_required_gib']==pytest.approx(29.1 if dtype=='float32' else 16.55)
    # Zero current-process allocations must not disguise other GPU memory use.
    assert report['current_process_allocated_gib']==report['current_process_reserved_gib']==0
    assert len(result['missing'])==int(bool(limits))
    if limits:
        message=result['missing'][0]
        for field in ('total=', 'free=', 'budget=', 'cuda:1', *limits):assert field in message


def test_no_cuda_preserves_blocked_status_and_skips_device_queries(monkeypatch):
    from cellquant import experiment
    config=json.loads(Path('configs/colab_pilot.json').read_text(encoding='utf-8'))
    monkeypatch.setattr(experiment,'environment_report',lambda:{'cuda':{'available':False}})
    def forbidden(*args,**kwargs):raise AssertionError('Must not query absent CUDA')
    monkeypatch.setattr(torch.cuda,'mem_get_info',forbidden)
    result=experiment.preflight(config)
    assert result['status']=='BLOCKED' and result['gpu_memory'] is None
    assert 'CUDA unavailable for model stages' in result['missing']
