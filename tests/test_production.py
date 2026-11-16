import math
import numpy as np
import pytest

from cellquant.events import GroupOrbit, WorkBudget, owner_intervals
from cellquant.quantization import QuantSpec, quantize, torch_quantize, risk
from cellquant.solver import (CalibrationPanel, PanelState, PanelOrbit, SolverConfig,
                              fit_shared_rotation, solve_pair, hadamard_rotation)
from cellquant.math_reference import PairProblem, solve_pair_reference, trig_value, givens


@pytest.mark.parametrize('placement',['input','output'])
@pytest.mark.parametrize('group',[2,4,-1])
@pytest.mark.parametrize('seed',[11,23,37])
def test_incremental_coefficients_match_direct(placement,group,seed):
    rng=np.random.default_rng(seed)
    d=4
    w=rng.normal(size=(2,d) if placement=='input' else (d,8))
    x=rng.normal(size=(w.shape[1],3))
    p=CalibrationPanel('p',placement,w,x)
    spec=QuantSpec(group_size=group)
    st=PanelState(p,np.eye(d),spec,spec)
    orb=PanelOrbit(st,(0,3))
    events={-.3,.3}
    for g in orb.groups:
        events.update(g.enumerate(-.3,.3,True,WorkBudget())[0])
    events=sorted(events)
    for lo,hi in zip(events[:-1],events[1:]):
        middle=(lo+hi)/2
        for k in range(len(orb.groups)):orb.replace(k,middle)
        for theta in [lo+(hi-lo)*.23,lo+(hi-lo)*.71]:
            direct=st.direct((0,3),theta)
            assert abs(trig_value(orb.loss_coeff,theta)-direct['loss'])<1e-10
            assert abs(trig_value(orb.bw,theta)-direct['budget_w'])<1e-11
            assert abs(trig_value(orb.ba,theta)-direct['budget_a'])<1e-11


@pytest.mark.parametrize('seed',[11,23,37])
@pytest.mark.parametrize('pair',[(0,1),(0,3)])
def test_solver_agrees_with_reference_and_screening(seed,pair):
    rng=np.random.default_rng(seed)
    w,x=rng.normal(size=(2,4)),rng.normal(size=(4,2))
    spec=QuantSpec(group_size=2)
    st=PanelState(CalibrationPanel('p','input',w,x),np.eye(4),spec,spec)
    limits={'budget_w':st.bw,'budget_a':st.ba}
    prod=solve_pair([st],pair,limits,SolverConfig(theta_limit=.3))
    full=solve_pair([st],pair,limits,SolverConfig(theta_limit=.3,screen=False))
    ref=solve_pair_reference(PairProblem(w,x,pair,group_size=2,left=-.3,right=.3))
    assert prod['best']['loss']==pytest.approx(ref['best']['loss'],rel=2e-6,abs=1e-10)
    assert prod['best']['loss']==pytest.approx(full['best']['loss'],rel=1e-7,abs=1e-10)
    assert prod['best']['budget_w']<=st.bw+1e-10
    assert prod['best']['budget_a']<=st.ba+1e-10


def test_shared_rotation_output_direction_and_resume(tmp_path):
    rng=np.random.default_rng(7);d=4
    panels=[CalibrationPanel('in','input',rng.normal(size=(2,d)),rng.normal(size=(d,3))),
            CalibrationPanel('down','output',rng.normal(size=(d,8)),rng.normal(size=(8,3)))]
    cfg=SolverConfig(theta_limit=.2,checkpoint_path=str(tmp_path/'state.npz'))
    result=fit_shared_rotation(panels,np.eye(d),QuantSpec(group_size=4),[(0,1),(1,3)],config=cfg)
    assert result['status']=='COMPLETE'
    assert result['final']['loss']<=result['initial']['loss']+1e-12
    assert result['orthogonality_residual']<1e-12
    cfg.resume=True
    resumed=fit_shared_rotation(panels,np.eye(d),QuantSpec(group_size=4),[(0,1),(1,3)],config=cfg)
    np.testing.assert_array_equal(resumed['q'],result['q'])
    panels[0].activation[0,0]+=1
    with pytest.raises(ValueError,match='mismatch'):
        fit_shared_rotation(panels,np.eye(d),QuantSpec(group_size=4),[(0,1),(1,3)],config=cfg)


def test_zero_groups_and_actual_nearest_even():
    z=np.array([[0.,0.,0.,0.],[7.,.5,1.5,-2.5]])
    deq,codes,scales=quantize(z,QuantSpec())
    np.testing.assert_array_equal(codes[1],[7,0,2,-2])
    assert scales[0,0,0]==0 and risk(scales[:1],4)==0
    w=np.zeros((2,4));x=np.zeros((4,2))
    r=fit_shared_rotation([CalibrationPanel('zero','input',w,x)],np.eye(4),QuantSpec(),[(0,1)])
    assert r['final']['loss']==r['final']['budget_w']==r['final']['budget_a']==0


def test_isolated_radius_tie_and_simultaneous_boundaries():
    # At theta=0 moving entries tie a constant envelope owner. Scale must stay constant.
    c=np.array([[0.,5.,0.],[0.,0.,5.],[5.,0.,0.],[.5,0.,0.]])
    g=GroupOrbit('w',0,0,c,QuantSpec(),True)
    np.testing.assert_array_equal(g.scale_coeff(0),[5/7,0,0])
    for theta in [0.,np.nextafter(0.,1.),-.1,.1]:
        actual,_=g.reconstruction(theta,actual=True)
        z=c@np.array([1.,math.cos(theta),math.sin(theta)])
        np.testing.assert_allclose(actual[:,0],quantize(z[None,:])[0][0])


def test_output_envelope_keeps_all_members_and_degenerate_cases():
    for points in [np.zeros((16,2)),np.c_[np.arange(16),np.zeros(16)],
                   np.random.default_rng(8).normal(size=(32,2))]:
        coeff=np.c_[np.zeros(len(points)),points]
        for lo,hi,owner,sign in owner_intervals(coeff,-.7,.7):
            for theta in np.linspace(lo,hi,7)[1:-1]:
                values=coeff@np.array([1,math.cos(theta),math.sin(theta)])
                assert abs(sign*values[owner]-np.max(np.abs(values)))<1e-12


def test_event_limit_cannot_report_success(tmp_path):
    rng=np.random.default_rng(9)
    panel=CalibrationPanel('p','input',rng.normal(size=(2,8)),rng.normal(size=(8,3)))
    cfg=SolverConfig(event_limit_per_pair=1,checkpoint_path=str(tmp_path/'blocked.npz'))
    r=fit_shared_rotation([panel],np.eye(8),QuantSpec(),[(0,1)],config=cfg)
    assert r['status']=='RESOURCE_BLOCKED' and r['completed_pairs']==0
    np.testing.assert_array_equal(r['q'],np.eye(8))


def test_fixed_scale_control_coefficients():
    rng=np.random.default_rng(12)
    panel=CalibrationPanel('p','output',rng.normal(size=(4,8)),rng.normal(size=(8,2)))
    state=PanelState(panel,np.eye(4),QuantSpec(),QuantSpec(),fixed=True)
    orb=PanelOrbit(state,(0,1));theta=.12
    for k in range(len(orb.groups)):orb.replace(k,theta)
    assert trig_value(orb.loss_coeff,theta)==pytest.approx(state.direct((0,1),theta)['loss'],abs=1e-12)


def test_unsupported_and_truncated_configs_rejected():
    with pytest.raises(NotImplementedError):quantize(np.ones((2,4)),QuantSpec(symmetric=False))
    with pytest.raises(NotImplementedError):quantize(np.ones((2,4)),QuantSpec(clip_ratio=.9))
    with pytest.raises(ValueError):CalibrationPanel('bad','input',np.ones((2,4)),np.ones((4,2)),complete_group_width=8)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_torch_codes_and_scales_match_numpy(device):
    torch=pytest.importorskip('torch')
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    z=np.random.default_rng(1).normal(size=(3,4096));z[0,:128]=0
    spec=QuantSpec(group_size=128)
    expected=quantize(z,spec)
    actual=torch_quantize(torch.tensor(z,device=device),spec)
    for a,b in zip(actual,expected):np.testing.assert_allclose(a.cpu().numpy(),b,rtol=1e-14,atol=1e-14)
def test_cuda_dense_shared_solver_matches_cpu():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable; no compatibility claim')
    from cellquant.solver import fit_shared_rotation,hadamard_rotation
    rng=np.random.default_rng(87)
    panels=[CalibrationPanel('input','input',rng.normal(size=(2,4)),rng.normal(size=(4,3))),
            CalibrationPanel('output','output',rng.normal(size=(4,8)),rng.normal(size=(8,3)))]
    q=hadamard_rotation(4,11);results=[]
    for device in ['cpu','cuda']:
        results.append(fit_shared_rotation(panels,q,QuantSpec(group_size=4),[(0,2)],config=SolverConfig(theta_limit=.1,device=device)))
    assert all(r['status']=='COMPLETE' for r in results)
    assert abs(results[0]['final']['loss']-results[1]['final']['loss'])<1e-9


def test_interrupt_between_q_write_and_log_keeps_last_transaction(tmp_path,monkeypatch):
    import cellquant.solver as solver
    from dataclasses import replace
    rng=np.random.default_rng(91)
    panels=[CalibrationPanel('i','input',rng.normal(size=(3,4)),rng.normal(size=(4,5)))]
    cfg=SolverConfig(theta_limit=.2,constrain_budgets=False,checkpoint_path=str(tmp_path/'state.npz'))
    expected=fit_shared_rotation(panels,np.eye(4),QuantSpec(),[(0,1)],config=replace(cfg,checkpoint_path=None))
    assert expected['factors']
    original=solver.rotate_rows;stop=[True]
    def interrupted(matrix,pair,theta):
        original(matrix,pair,theta)
        if matrix.shape==(4,4) and stop[0]:stop[0]=False;raise KeyboardInterrupt('mid-transaction fixture')
    monkeypatch.setattr(solver,'rotate_rows',interrupted)
    with pytest.raises(KeyboardInterrupt):fit_shared_rotation(panels,np.eye(4),QuantSpec(),[(0,1)],config=cfg)
    actual=fit_shared_rotation(panels,np.eye(4),QuantSpec(),[(0,1)],config=replace(cfg,resume=True))
    np.testing.assert_array_equal(actual['q'],expected['q'])
    assert actual['factors']==expected['factors']
