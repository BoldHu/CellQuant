import json
from pathlib import Path
import numpy as np
import pytest
from cellquant.data import (split_train_blocks,verify_disjoint,token_hash,save_token_manifest,load_token_manifest,wikitext_documents)
from cellquant.solver import CalibrationPanel,SolverConfig
from cellquant.quantization import QuantSpec
from cellquant.baselines import fit_baseline,METHODS
from cellquant.experiment import preflight


def blocks(n):
    return [{'document_id':str(i),'document_sha256':str(i),'block_id':f'{i}:0','start_token':0,
             'tokens':np.arange(8)+i,'token_ids_sha256':token_hash(np.arange(8)+i)} for i in range(n)]


def test_document_split_and_hash_tampering(tmp_path):
    fit,dev=split_train_blocks(blocks(20),8,8,11);assert verify_disjoint(fit,dev)
    with pytest.raises(ValueError):verify_disjoint(fit,fit)
    with pytest.raises(ValueError):split_train_blocks(blocks(4),3,3,11)
    path=save_token_manifest(tmp_path,{'fit':fit,'dev':dev},{'source':'test'},source_type='unit_test_fixture')
    with pytest.raises(ValueError):load_token_manifest(path)
    manifest,parts=load_token_manifest(path,allow_fixture=True)
    assert len(parts['fit'])==8
    (tmp_path/'token_ids.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):load_token_manifest(path,allow_fixture=True)


def test_wikitext_articles_not_paragraphs():
    docs=list(wikitext_documents([' = First = ','paragraph',' == subsection == ','body',' = Second = ','next']))
    assert len(docs)==2 and 'subsection' in docs[0][1]


@pytest.mark.parametrize('method',METHODS)
def test_all_matched_methods_preserve_constraints(method):
    pytest.importorskip('torch')
    rng=np.random.default_rng(19)
    panels=[CalibrationPanel('i','input',rng.normal(size=(2,4)),rng.normal(size=(4,3))),
            CalibrationPanel('o','output',rng.normal(size=(4,8)),rng.normal(size=(8,3)))]
    result=fit_baseline(method,panels,np.eye(4),QuantSpec(group_size=4),[(0,1),(1,3)],
                        SolverConfig(theta_limit=.2,wall_seconds=30.),ste_steps=8,grid_points=9)
    assert result['status']=='COMPLETE'
    assert result['final']['loss']<=result['initial']['loss']+1e-12
    assert result['final']['budget_w']<=result['limits']['budget_w']+1e-10
    assert result['final']['budget_a']<=result['limits']['budget_a']+1e-10
    assert result['orthogonality_residual']<1e-12


def test_backend_is_real_callable_but_templates_stay_blocked():
    from cellquant.runner import backend_available
    assert backend_available()
    config=json.loads(Path('configs/jobs/canary_llama2_7b_cellquant_s11.json').read_text(encoding='utf-8'))
    ready=preflight(config)
    assert ready['status']=='BLOCKED'
    assert any('G1' in r for r in ready['missing'])
    assert any('revision' in r for r in ready['missing'])


def test_export_requires_all_r2_and_correct_orientation(tmp_path):
    torch=pytest.importorskip('torch')
    from cellquant.spinquant_adapter import export_spinquant_rotation
    from cellquant.math_reference import givens
    q=givens(4,(0,1),.2)
    with pytest.raises(ValueError):export_spinquant_rotation(q,[np.eye(2)],tmp_path/'bad.bin',num_layers=2)
    path=tmp_path/'R.bin';export_spinquant_rotation(q,[np.eye(2),np.eye(2)],path,num_layers=2)
    state=torch.load(path,weights_only=True)
    np.testing.assert_allclose(state['R1'].numpy(),q.T,atol=1e-7)
    assert len(state)==3


@pytest.mark.parametrize('method',['cellquant','givens_ste_matched','angle_grid_matched'])
def test_location_specific_complete_groups(method):
    pytest.importorskip('torch')
    rng=np.random.default_rng(73)
    panels=[CalibrationPanel('q_proj','input',rng.normal(size=(2,4)),rng.normal(size=(4,3))),
            CalibrationPanel('o_proj','output',rng.normal(size=(4,4)),rng.normal(size=(4,3)))]
    specs={'q_proj':(QuantSpec(group_size=4),QuantSpec(group_size=4)),
           'o_proj':(QuantSpec(group_size=4),QuantSpec(group_size=2))}
    r=fit_baseline(method,panels,np.eye(4),specs,[(0,3)],SolverConfig(theta_limit=.1),ste_steps=3,grid_points=5)
    assert r['final']['loss']<=r['initial']['loss']+1e-12
    assert r['final']['budget_w']<=r['limits']['budget_w']+1e-10


@pytest.mark.parametrize('backend',['numpy_reference','torch'])
@pytest.mark.parametrize('method',['angle_grid_matched','givens_ste_matched'])
def test_control_hitting_method_budget_does_not_report_complete(backend,method):
    pytest.importorskip('torch')
    panel=CalibrationPanel('in','input',np.ones((2,4)),np.ones((4,2)))
    result=fit_baseline(method,[panel],np.eye(4),QuantSpec(),[(0,1)],
                        SolverConfig(backend=backend,device='cpu',wall_seconds=0))
    assert result['status']=='RESOURCE_BLOCKED' and result['reason']
    assert result['completed_pairs']==0 and result['requested_pairs']==1


@pytest.mark.parametrize('profile_only,fit_only', [(True,False),(False,False),(False,True)])
def test_prepared_capture_visits_only_consumed_splits(profile_only,fit_only,monkeypatch):
    pytest.importorskip('torch')
    from types import SimpleNamespace
    from cellquant import pilot, spinquant_adapter
    calls=[]
    model=SimpleNamespace(config=SimpleNamespace(hidden_size=4),parameters=lambda: [])
    parts={'fit':blocks(2)[:1],'dev':blocks(2)[1:]}
    monkeypatch.setattr(pilot,'load_token_manifest',lambda path: ({},parts))
    monkeypatch.setattr(spinquant_adapter,'panel_quant_specs',lambda *args: QuantSpec())
    def capture(model,partition,**kwargs):
        calls.append((kwargs['seed'],partition[0]['document_id'],kwargs['tokens_per_panel']))
        return [kwargs['seed']]
    monkeypatch.setattr(pilot,'capture_panels',capture)
    config={'runtime':{'model_device':'cpu'},'seeds':[11,23],'quantization':{},
            'data':{'token_manifests':{'11':'first','23':'second'}},
            'formal_pilot':{'layers':[0],'projections':['q_proj'],'input_weight_rows':64,
                            'fit_tokens_per_panel':128,'heldout_tokens_per_panel':256}}
    prepared=pilot.prepare_pilot_inputs(config,model,graph_identity='fixture',profile_only=profile_only,fit_only=fit_only)
    seeds=[11] if profile_only else [11,23]
    splits=[('0',128)] if profile_only or fit_only else [('0',128),('1',256)]
    assert calls==[(s,doc,n) for s in seeds for doc,n in splits]
    assert list(prepared['seeds'])==seeds
    assert ('heldout' in prepared['seeds'][11]) == (not (profile_only or fit_only))


def test_frozen_model_solver_selects_matching_backend_and_checkpoint(tmp_path):
    from cellquant.experiment import _model_solver_config
    for name in ['autodl_canary','full_experiments']:
        settings=json.loads(Path(f'configs/{name}.json').read_text(encoding='utf8'))['solver']
        cfg=_model_solver_config({**settings,'cell_batch_size':37},tmp_path,'fingerprint')
        assert (cfg.backend,cfg.device,cfg.compute_dtype)==('torch','cuda:0','float64')
        assert cfg.cell_batch_size==37 and Path(cfg.checkpoint_path).suffix=='.pt'
    legacy=_model_solver_config({},tmp_path,'fingerprint')
    assert (legacy.backend,legacy.device,Path(legacy.checkpoint_path).suffix)==('numpy_reference','cpu','.npz')


def test_fp_audit_scores_next_tokens_and_rejects_nonfinite_logits():
    torch=pytest.importorskip('torch')
    from cellquant.spinquant_adapter import compare_logits
    reference=torch.zeros((1,3,4),dtype=torch.float32)
    actual=reference.clone();actual[0,0,2]=1;actual[0,1,3]=2;actual[0,2]=99
    ids=torch.tensor([[1,2,3]])
    delta=compare_logits(reference,actual,token_ids=ids)
    expected=(np.log(np.exp(1)+3)-1+np.log(np.exp(2)+3)-2)/2-np.log(4)
    assert delta['mean_nll_delta']==pytest.approx(expected)
    actual[0,0,0]=torch.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        compare_logits(reference,actual,token_ids=ids)
