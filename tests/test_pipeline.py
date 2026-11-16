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
