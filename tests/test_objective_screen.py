from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from cellquant import pilot
from cellquant.baselines import METHODS, fit_baseline
from cellquant.data import confirmation_partitions
from cellquant.objective_screen import (ARMS, CELL, GRID, HAD, NEW, OLD, assess_screen,
                                        screen_seed, verified_screen, verify_parent_manifests)
from cellquant.quantization import QuantSpec
from cellquant.runtime import atomic_json, sha256_file
from cellquant.solver import CalibrationPanel, PanelState, SolverConfig, checkpoint_config, configure_fit_objective
from cellquant.tensor_solver import hadamard_rotation


def panels(seed, label):
    rng=np.random.default_rng(seed)
    def panel(name,side,w,x):
        return CalibrationPanel(name,side,w,x,metadata={'source_type':'unit_test_fixture','layer':0,
            'document_ids':[label], 'block_ids':[label+':0'], 'tokens_per_block':[2]})
    return [panel('q','input',rng.normal(size=(3,4)),rng.normal(size=(4,2))),
            panel('down','output',rng.normal(size=(4,8)),rng.normal(size=(8,2)))]


def test_objective_screen_binds_both_parent_token_manifests(tmp_path):
    manifests = {s: atomic_json(tmp_path / (s+'.json'), {'seed':s}) for s in ('11','23')}
    protocol = atomic_json(tmp_path/'protocol.json', {'token_manifest_sha256':{s:sha256_file(p) for s,p in manifests.items()}})
    cfg = {'seeds':[11,23], 'protocol_manifest_path':protocol, 'protocol_sha256':sha256_file(protocol),
           'data':{'token_manifests':manifests}}
    verify_parent_manifests(cfg)
    atomic_json(manifests['23'], {'seed':'changed'})
    with pytest.raises(ValueError,match='token manifests'):
        verify_parent_manifests(cfg)


def test_frozen_weights_change_only_fit_loss_and_preserve_default_checkpoint():
    fit=panels(5,'a')
    spec=QuantSpec(group_size=4)
    q=np.eye(4)
    raw=[PanelState(p,q,spec,spec) for p in fit]
    weighted=[PanelState(p,q,spec,spec) for p in fit]
    record=configure_fit_objective(weighted,NEW)
    assert sum(s.loss for s in weighted)==pytest.approx(sum(s.loss for s in raw))
    assert weighted[0].loss==pytest.approx(weighted[1].loss)
    for s,t in zip(raw,weighted):
        w=record['loss_multipliers'][s.panel.panel_id]
        a,b=s.direct((0,1),.18),t.direct((0,1),.18)
        assert b['loss']==pytest.approx(w*a['loss'])
        assert all(a[k]==b[k] for k in ('budget_w','budget_a'))
        t.accept((0,1),.18)
        assert t.loss==pytest.approx(b['loss'])
    assert 'fit_objective' not in checkpoint_config(SolverConfig())
    assert checkpoint_config(SolverConfig(fit_objective=NEW))['fit_objective']==NEW
    zero=CalibrationPanel('zero','input',np.zeros((2,4)),np.ones((4,2)))
    state=PanelState(zero,q,spec,spec)
    assert configure_fit_objective([state],NEW)['loss_multipliers']=={'zero':1.}
    with pytest.raises(ValueError,match='Unknown'):
        configure_fit_objective(raw,'typo')


@pytest.mark.parametrize('backend,device',[('numpy_reference','cpu'),('torch','cpu'),
    pytest.param('torch','cuda:0',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='no CUDA'))])
def test_all_five_methods_share_weights_risks_and_direct_fit_loss(backend,device):
    spec=QuantSpec(group_size=4)
    fit=panels(7,'fit')
    cfg=SolverConfig(backend=backend,device=device,fit_objective=NEW,theta_limit=.12,
                     wall_seconds=120,seconds_per_pair=45)
    q=np.eye(4)
    policies=[]
    for m in METHODS:
        result=fit_baseline(m,fit,q,spec,[(0,1)],cfg,grid_points=9,ste_steps=3)
        assert result['status']=='COMPLETE'
        policies.append(result['fit_objective'])
        if backend=='torch':
            from cellquant.tensor_solver import PanelState as State
            q0=torch.as_tensor(q,dtype=torch.float64,device=device)
        else:
            State,q0=PanelState,q
        states=[State(p,q0,spec,spec,fixed=m=='fixed_scale_search_matched') for p in fit]
        assert result['limits']==pytest.approx({'budget_w':sum(float(s.bw) for s in states),
                                               'budget_a':sum(float(s.ba) for s in states)})
        for factor in result['factors']:
            for state in states:
                state.accept(tuple(factor['pair']),factor['theta'])
        expected=sum(float(s.loss)*policies[-1]['loss_multipliers'][s.panel.panel_id] for s in states)
        assert result['final']['loss']==pytest.approx(expected,rel=1e-9,abs=1e-12)
    assert all(p==policies[0] for p in policies)


def test_screen_seed_persists_paired_raw_metrics_and_never_generates_g1(tmp_path):
    cfg={'seeds':[11,23],'solver':{'backend':'torch','device':'cpu','theta_limit':.12,
         'wall_seconds_per_method':120,'seconds_per_pair':40}}
    ready={'fit':panels(5,'fit'),'screen_validation':panels(6,'val'),'specs':QuantSpec(group_size=4),'parent_manifest_sha256':'fixture'}
    rows,refs,reason=screen_seed(ready,hadamard_rotation(4,11,'cpu'),[(0,1)],cfg,11,tmp_path,'fixture')
    assert reason is None and len(refs)==len(rows)==5
    assert all(sha256_file(ref['path'])==ref['sha256'] for ref in refs)
    for row in rows:
        assert row['fit_documents']==['fit'] and row['validation_documents']==['val']
        assert row['source_type']=='unit_test_fixture' and len(row['validation'])==2
        for v in row['validation']:
            b,=v['blocks']
            assert v['nmse']==pytest.approx(b['squared_error']/b['target_energy'])
    assert not list(tmp_path.rglob('G1*'))
    assert assess_screen(rows)['decision'] in ('TEST_ONE_G1','STOP')


def screen_rows(gains=(.97,.97)):
    return [{'objective':mode,'method':method,'seed':seed,'validation':[
        {'panel_id':p,'nmse':gains[i] if (mode,method)==(NEW,CELL) else
         (1.03 if (mode,method)==(NEW,GRID) else 1.)} for p in ('q','down')]}
        for i,seed in enumerate((11,23)) for mode,method in ((OLD,HAD),*ARMS)]


def test_screen_uses_stronger_grid_both_seeds_and_original_cell():
    assert assess_screen(screen_rows())['decision']=='TEST_ONE_G1'
    assert assess_screen(screen_rows((.94,1.01)))['decision']=='STOP'
    assert assess_screen(screen_rows((.996,.996)))['decision']=='STOP'
    rows=screen_rows()
    for row in rows:
        if (row['objective'],row['method'])==(OLD,CELL):
            for v in row['validation']:v['nmse']=.94
    assert assess_screen(rows)['decision']=='STOP'
    with pytest.raises(ValueError,match='Missing'):
        assess_screen(rows[:-1])
    rows[0]['validation'][0]['nmse']=0.
    assert assess_screen(rows)['decision']=='INSUFFICIENT_EVIDENCE'


def test_fresh_documents_exclude_previous_ids_content_and_duplicate_token_blocks():
    def block(n,doc=None,tokens=None):
        return {'document_id':str(n),'document_sha256':str(doc or n),'token_ids_sha256':str(tokens or n)}
    old={11:{'fit':[block(1)],'dev':[block(2)]},23:{'fit':[block(3)],'dev':[block(4)]}}
    pool=[block(i) for i in range(1,45)]+[block(50,1),block(51,100,2),block(52,30)]
    chosen=confirmation_partitions(pool,old,8)
    assert chosen==confirmation_partitions(pool,old,8)
    for s,(fit,_) in chosen.items():
        assert fit==old[s]['fit']
    for key in ('document_id','document_sha256','token_ids_sha256'):
        vals=[b[key] for _,docs in chosen.values() for b in docs]
        assert len(set(vals))==16 and not set(vals)&{'1','2','3','4'}
    with pytest.raises(ValueError,match='Insufficient'):
        confirmation_partitions(pool[:8],old,8)


def test_unverified_summary_cannot_prepare_a_confirmation(tmp_path):
    atomic_json(tmp_path/'result.json',{'status':'SCREEN_COMPLETE','source_type':'unit_test_fixture',
                                        'objective_screen':{'decision':'TEST_ONE_G1'}})
    with pytest.raises(ValueError,match='real screen'):
        verified_screen(tmp_path,{})


def test_capture_holds_out_whole_fit_documents_and_never_scores_old_dev(tmp_path,monkeypatch):
    from cellquant import spinquant_adapter
    path=atomic_json(tmp_path/'source.json',{'source_type':'unit_test_fixture'})
    def docs(prefix):
        return [{'document_id':prefix+str(i),'document_sha256':prefix+str(i)} for i in range(8)]
    parts={'fit':docs('fit'),'dev':docs('dev')}
    monkeypatch.setattr(pilot,'load_token_manifest',lambda _ : ({},parts))
    monkeypatch.setattr(spinquant_adapter,'panel_quant_specs',lambda *args: QuantSpec())
    calls=[]
    def capture(model,docs,**kw):
        calls.append(([b['document_id'] for b in docs],kw['tokens_per_panel']))
        return []
    monkeypatch.setattr(pilot,'capture_panels',capture)
    config={'seeds':[11],'runtime':{'model_device':'cpu'},'quantization':{},
        'data':{'token_manifests':{'11':str(path)}},'formal_pilot':{'layers':[0],'projections':['q_proj'],
        'input_weight_rows':64,'fit_tokens_per_panel':128,'heldout_tokens_per_panel':256}}
    model=SimpleNamespace(config=SimpleNamespace(hidden_size=4),parameters=lambda:[])
    out=pilot.prepare_pilot_inputs(config,model,graph_identity='fixture',screen_objective=True)
    assert calls==[([f'fit{i}' for i in range(4)],128),([f'fit{i}' for i in range(4,8)],256)]
    assert 'screen_validation' in out['seeds'][11] and 'heldout' not in out['seeds'][11]
    parts['fit'][6]['document_sha256']=parts['fit'][0]['document_sha256']
    with pytest.raises(ValueError,match='eight distinct'):
        pilot.prepare_pilot_inputs(config,model,graph_identity='fixture',screen_objective=True)
