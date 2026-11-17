"""Independent direct-quantizer checks of batched event-cell objectives."""
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from cellquant import tensor_solver as ts
from cellquant.tensor_cells import output_terms,input_term
from cellquant.tensor_math import trig_value
from cellquant.events import WorkBudget
from cellquant.quantization import QuantSpec
from cellquant.solver import CalibrationPanel,SolverConfig,PanelState,solve_pair


@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    return request.param


def evaluate_terms(terms,theta):
    result=theta.new_zeros(3)
    for t in terms:
        ix=(torch.searchsorted(t.angles,theta,right=True)-1).clamp(0,len(t.loss)-1)
        val=torch.stack([trig_value(c[ix],theta) for c in (t.loss,t.bw,t.ba)])
        boundary=torch.searchsorted(t.angles,theta)
        if boundary<len(t.angles) and bool(t.angles[boundary]==theta):val=t.boundary[boundary]
        result+=t.weight*val
    return result


@pytest.mark.parametrize('group_size',[-1,4])
@pytest.mark.parametrize('fixed',[False,True])
def test_output_prefix_matches_direct_quantization_and_chunking(device,group_size,fixed):
    rng=np.random.default_rng(573)
    p=CalibrationPanel('out','output',rng.normal(size=(8,24)),rng.normal(size=(24,5)))
    spec=QuantSpec(group_size=group_size);q=torch.eye(8,dtype=torch.float64,device=device)
    s=ts.PanelState(p,q,spec,spec,fixed=fixed);pair=(1,6)
    work=WorkBudget(200000,120)
    o=ts.PanelOrbit(s,pair);events=[g.enumerate(-.78,.78,True,work)[:2] for g in o.groups]
    small=output_terms(o,events,work,3,lambda *a,**k:None)
    large=output_terms(o,events,work,1024,lambda *a,**k:None)
    boundaries=torch.unique(torch.cat([t.angles for t in small]),sorted=True)
    samples=torch.unique(torch.linspace(0,len(boundaries)-1,min(len(boundaries),45),device=device).long())
    for i in samples:
        theta=boundaries[i]
        direct=s.direct(pair,theta)
        expected=torch.stack([direct[k] for k in ('loss','budget_w','budget_a')])
        torch.testing.assert_close(evaluate_terms(small,theta),expected,atol=2e-11,rtol=2e-10)
        torch.testing.assert_close(evaluate_terms(large,theta),expected,atol=2e-11,rtol=2e-10)
        if i<len(boundaries)-1:
            theta=theta+(boundaries[i+1]-theta)*.37
            direct=s.direct(pair,theta);expected=torch.stack([direct[k] for k in ('loss','budget_w','budget_a')])
            torch.testing.assert_close(evaluate_terms(small,theta),expected,atol=2e-11,rtol=2e-10)
    assert o.update_stats['full_gemm_per_event']==0
    assert o.update_stats['group_projection_rebases']>0


@pytest.mark.parametrize('fixed',[False,True])
def test_shared_batched_angle_matches_independent_cpu_solver(device,fixed):
    rng=np.random.default_rng(563)
    panels=[CalibrationPanel('in','input',rng.normal(size=(2,8)),rng.normal(size=(8,3)),objective_weight=.7),
            CalibrationPanel('out','output',rng.normal(size=(8,12)),rng.normal(size=(12,3)),objective_weight=1.3)]
    spec=QuantSpec(group_size=4);q=torch.eye(8,dtype=torch.float64,device=device);pair=(0,5)
    cpu=[PanelState(p,np.eye(8),spec,spec,fixed=fixed) for p in panels]
    gpu=[ts.PanelState(p,q,spec,spec,fixed=fixed) for p in panels]
    limits={k:v for k,v in ts.aggregate(gpu).items() if k!='loss'}
    cfg=SolverConfig(theta_limit=.36,seconds_per_pair=120,backend='torch',device=device,cell_batch_size=7)
    expected=solve_pair(cpu,pair,{k:float(v) for k,v in limits.items()},cfg)
    actual=ts.solve_pair(gpu,pair,limits,cfg)
    assert actual['best']['loss']==pytest.approx(expected['best']['loss'],rel=1e-7,abs=1e-10)
    assert actual['best']['budget_w']<=float(limits['budget_w'])+1e-10
    assert actual['best']['budget_a']<=float(limits['budget_a'])+1e-10
    assert actual['cell_engine']=='torch_batched_bilateral_prefix_v2'
    assert actual['boundary_rounding_evaluations']==actual['event_count']


def test_zero_and_simultaneous_output_boundaries(device):
    spec=QuantSpec(group_size=4)
    for zero in [True,False]:
        w=np.zeros((4,8)) if zero else np.tile([1.,-1.,.5,-.5,2.,-2.,.5,-.5],(4,1))
        p=CalibrationPanel('ties','output',w,np.tile([1.,2.],(8,1)))
        s=ts.PanelState(p,torch.eye(4,dtype=torch.float64,device=device),spec,spec)
        o=ts.PanelOrbit(s,(0,1));work=WorkBudget(200000,60)
        events=[g.enumerate(-.7,.7,True,work)[:2] for g in o.groups]
        terms=output_terms(o,events,work,5,lambda *a,**k:None)
        for theta in torch.unique(torch.cat([t.angles for t in terms])):
            direct=s.direct((0,1),theta)
            torch.testing.assert_close(evaluate_terms(terms,theta),torch.stack([direct[k] for k in ('loss','budget_w','budget_a')]),rtol=1e-10,atol=1e-11)


def test_all_zero_panels_and_invalid_batch(device):
    spec=QuantSpec(group_size=4)
    panels=[CalibrationPanel('zero.in','input',np.zeros((2,4)),np.zeros((4,2))),
            CalibrationPanel('zero.out','output',np.zeros((4,8)),np.zeros((8,2)))]
    q=torch.eye(4,dtype=torch.float64,device=device)
    cfg=SolverConfig(backend='torch',device=device,seconds_per_pair=60,cell_batch_size=2)
    r=ts.fit_baseline('cellquant',panels,q,spec,[(0,1)],cfg)
    assert r['status']=='COMPLETE' and r['completed_pairs']==1 and r['final']['loss']==0
    torch.testing.assert_close(r['q'],q,atol=0,rtol=0)
    cfg.cell_batch_size=0
    with pytest.raises(ValueError,match='cell_batch_size'):ts.fit_baseline('cellquant',panels,q,spec,[(0,1)],cfg)


@pytest.mark.parametrize('group_size',[-1,4])
@pytest.mark.parametrize('fixed',[False,True])
def test_radius_stable_input_rank_two_preserves_full_group_objective(device,group_size,fixed):
    rng=np.random.default_rng(665)
    w=rng.normal(size=(3,8));x=rng.normal(size=(8,5))
    w[:,[0,4]]*=.1;x[[0,4]]*=.1
    w[:,[2,6]]=4.;x[[2,6]]=4.
    p=CalibrationPanel('stable','input',w,x);spec=QuantSpec(group_size=group_size)
    s=ts.PanelState(p,torch.eye(8,dtype=torch.float64,device=device),spec,spec,fixed=fixed)
    o=ts.PanelOrbit(s,(0,4));assert all(g.stable for g in o.groups)
    work=WorkBudget(200000,60);events=[g.enumerate(-.78,.78,True,work)[:2] for g in o.groups]
    term=input_term(o,events,work,lambda *a,**k:None,3)
    assert o.update_stats['constant_scale_input_rank_two_cells']==len(term.loss)
    for theta in torch.cat((term.angles,term.angles[:-1]+(term.angles[1:]-term.angles[:-1])*.39)):
        direct=s.direct((0,4),theta)
        torch.testing.assert_close(evaluate_terms([term],theta),torch.stack([direct[k] for k in ('loss','budget_w','budget_a')]),rtol=1e-10,atol=1e-11)


def test_fixed_decoder_still_tracks_changing_physical_scale_risk(device):
    rng=np.random.default_rng(886)
    w=rng.normal(size=(3,8));x=rng.normal(size=(8,5));w[:,0]*=4.;x[0]*=4.
    p=CalibrationPanel('fixed','input',w,x);spec=QuantSpec(group_size=4)
    s=ts.PanelState(p,torch.eye(8,dtype=torch.float64,device=device),spec,spec,fixed=True)
    o=ts.PanelOrbit(s,(0,5));assert any(not g.stable for g in o.groups)
    work=WorkBudget(200000,60);events=[g.enumerate(-.78,.78,True,work)[:2] for g in o.groups]
    term=input_term(o,events,work,lambda *a,**k:None,5)
    assert 'constant_scale_input_rank_two_cells' in o.update_stats
    for theta in torch.cat((term.angles,term.angles[:-1]+(term.angles[1:]-term.angles[:-1])*.39)):
        direct=s.direct((0,5),theta)
        torch.testing.assert_close(evaluate_terms([term],theta),torch.stack([direct[k] for k in ('loss','budget_w','budget_a')]),rtol=1e-10,atol=1e-11)


def test_unstable_full_row_input_boundaries_match_direct_bilateral_quantizer(device):
    rng=np.random.default_rng(137)
    w=rng.normal(size=(3,8));x=rng.normal(size=(8,5));w[:,0]*=4.;x[0]*=4.
    p=CalibrationPanel('unstable','input',w,x);spec=QuantSpec(group_size=-1)
    s=ts.PanelState(p,torch.eye(8,dtype=torch.float64,device=device),spec,spec)
    o=ts.PanelOrbit(s,(0,5));assert any(not g.stable for g in o.groups)
    work=WorkBudget(200000,60);events=[g.enumerate(-.78,.78,True,work)[:2] for g in o.groups]
    term=input_term(o,events,work,lambda *a,**k:None,5)
    assert o.update_stats['pointwise_boundary_updates']>0
    for theta in torch.cat((term.angles,term.angles[:-1]+(term.angles[1:]-term.angles[:-1])*.39)):
        direct=s.direct((0,5),theta)
        torch.testing.assert_close(evaluate_terms([term],theta),torch.stack([direct[k] for k in ('loss','budget_w','budget_a')]),rtol=1e-10,atol=1e-11)
