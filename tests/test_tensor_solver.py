"""CPU oracle vs actual torch CPU/CUDA math; synthetic tests, never G1 evidence."""
import math
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from cellquant import tensor_math as tm, tensor_solver as ts
from cellquant.events import owner_intervals as cpu_intervals, affine_roots as cpu_affine
from cellquant.math_reference import _trig_roots, trig_value
from cellquant.solver import CalibrationPanel,SolverConfig,PanelState,PanelOrbit,solve_pair
from cellquant.quantization import QuantSpec


@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.backends.cuda.matmul.allow_tf32=False
    return request.param


def tensor(a,device):return torch.as_tensor(a,dtype=torch.float64,device=device)


@pytest.mark.parametrize('size',[2,8,64])
def test_gpu_support_envelope(device,size):
    rng=np.random.default_rng(184+size);coeff=np.zeros((size,3));coeff[:,1:]=rng.normal(size=(size,2))
    intervals=tm.owner_intervals(tensor(coeff,device),-.78,.78)
    angles=np.linspace(-.779,.779,401)
    for lo,hi,owner,sign in intervals:
        lo,hi,i,s=float(lo),float(hi),int(owner),float(sign)
        selected=angles[(angles>lo)&(angles<hi)]
        if not len(selected):continue
        values=coeff[:,1,None]*np.cos(selected)+coeff[:,2,None]*np.sin(selected)
        np.testing.assert_allclose(s*values[i],np.max(np.abs(values),axis=0),atol=1e-12)
    assert float(intervals[0][0])==-.78 and float(intervals[-1][1])==.78


@pytest.mark.parametrize('coeff',[
    [[0,0,0],[0,0,0],[0,0,0]], [[0,1,0],[0,-2,0],[0,3,0]],
    [[1,0,0],[0,1,0],[0,0,1]], [[0,1,0],[0,1,0],[0,-1,0]],
])
def test_zero_tie_collinear_envelopes(device,coeff):
    z=tensor(coeff,device)
    for lo,hi,i,sign in tm.owner_intervals(z,-.7,.7):
        points=torch.linspace(float(lo),float(hi),23,device=device,dtype=torch.float64)
        vals=z[:,0,None]+z[:,1,None]*points.cos()+z[:,2,None]*points.sin()
        torch.testing.assert_close(sign*vals[i],vals.abs().amax(0),rtol=1e-12,atol=1e-12)


def test_gpu_polynomial_and_affine_roots(device):
    rng=np.random.default_rng(832)
    for degree in [1,2,4]:
        for _ in range(10):
            c=rng.normal(size=2*degree+1)
            expected=_trig_roots(c,-.78,.78)
            actual=tm.trig_roots(tensor(c,device),-.78,.78).cpu().numpy()
            np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=1e-7)
    assert not tm.trig_roots(tensor([0.,0.,0.],device),-.78,.78).numel()
    z=rng.normal(size=(100,3))
    expected=sorted(cpu_affine(z,-.78,.78)[0])
    actual=torch.sort(tm.affine_roots(tensor(z,device),-.78,.78)[0]).values.cpu().numpy()
    np.testing.assert_allclose(actual,expected,atol=2e-12)


def test_roots_never_call_eigvals_cpu_fallback(device,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('CPU eigvals path is forbidden')
    monkeypatch.setattr(torch.linalg,'eigvals',forbidden)
    roots=tm.trig_roots(tensor([0.,0.,1.,.2,-.3,.1,.2,.1,.1],device),-.78,.78)
    assert roots.device.type==device and roots.numel()>0


@pytest.mark.parametrize('placement',['input','output'])
def test_tensor_state_and_incremental_coefficients(device,placement):
    rng=np.random.default_rng(477);w=rng.normal(size=(3,4) if placement=='input' else (4,8));x=rng.normal(size=(w.shape[1],3))
    p=CalibrationPanel('fixture',placement,w,x);spec=QuantSpec(group_size=4)
    cpu=PanelState(p,np.eye(4),spec,spec);gpu=ts.PanelState(p,tensor(np.eye(4),device),spec,spec)
    assert gpu.a.device.type==device
    for theta in [-.11,0.,.09]:
        for k,v in gpu.direct((0,1),theta).items():assert float(v)==pytest.approx(cpu.direct((0,1),theta)[k],abs=1e-12)
        orbit=ts.PanelOrbit(gpu,(0,1));reference=PanelOrbit(cpu,(0,1))
        for gid in range(len(orbit.groups)):
            orbit.replace(gid,theta);reference.replace(gid,theta)
        np.testing.assert_allclose(orbit.loss_coeff.cpu(),reference.loss_coeff,atol=1e-11)
        np.testing.assert_allclose(orbit.bw.cpu(),reference.bw,atol=1e-11)
        assert float(tm.trig_value(orbit.loss_coeff,theta))==pytest.approx(cpu.direct((0,1),theta)['loss'],abs=1e-11)


def test_gpu_shared_pair_against_cpu(device):
    rng=np.random.default_rng(51);spec=QuantSpec(group_size=4)
    panels=[CalibrationPanel('in','input',rng.normal(size=(2,4)),rng.normal(size=(4,2))),
            CalibrationPanel('out','output',rng.normal(size=(4,8)),rng.normal(size=(8,2)))]
    cpu=[PanelState(p,np.eye(4),spec,spec) for p in panels]
    gpu=[ts.PanelState(p,tensor(np.eye(4),device),spec,spec) for p in panels]
    limits={k:v for k,v in ts.aggregate(gpu).items() if k!='loss'}
    config=SolverConfig(theta_limit=.08,seconds_per_pair=120.,device=device,backend='torch')
    expected=solve_pair(cpu,(0,1),{k:float(v) for k,v in limits.items()},config)
    actual=ts.solve_pair(gpu,(0,1),limits,config)
    assert actual['best']['loss']==pytest.approx(expected['best']['loss'],rel=1e-7,abs=1e-10)
    assert actual['best']['budget_w']<=float(limits['budget_w'])+1e-10
    assert all(s['full_gemm_per_event']==0 for s in actual['incremental_statistics'])


@pytest.mark.parametrize('method',['hadamard_shared','cellquant','fixed_scale_search_matched','angle_grid_matched','givens_ste_matched'])
def test_gpu_controls_keep_shared_q_and_budgets(device,method):
    rng=np.random.default_rng(123)
    panels=[CalibrationPanel('in','input',tensor(rng.normal(size=(2,4)),device),tensor(rng.normal(size=(4,2)),device)),
            CalibrationPanel('out','output',tensor(rng.normal(size=(4,8)),device),tensor(rng.normal(size=(8,2)),device))]
    config=SolverConfig(theta_limit=.025,seconds_per_pair=120,wall_seconds=120,device=device,backend='torch')
    if method=='cellquant':
        from cellquant.solver import fit_shared_rotation
        result=fit_shared_rotation(panels,torch.eye(4,device=device,dtype=torch.float64),QuantSpec(group_size=4),[(0,1)],config=config)
    else:
        result=ts.fit_baseline(method,panels,torch.eye(4,device=device,dtype=torch.float64),QuantSpec(group_size=4),[(0,1)],config,grid_points=7,ste_steps=3)
    assert result['q'].device.type==device
    assert result['status']=='COMPLETE'
    assert result['final']['loss']<=result['initial']['loss']+1e-12
    assert result['final']['budget_w']<=result['limits']['budget_w']+1e-10
    assert result['final']['budget_a']<=result['limits']['budget_a']+1e-10
    assert result['orthogonality_residual']<1e-12


def test_tensor_resume_and_resource_blocking(device,tmp_path):
    rng=np.random.default_rng(94)
    panels=[CalibrationPanel('i','input',rng.normal(size=(2,4)),rng.normal(size=(4,2))),
            CalibrationPanel('o','output',rng.normal(size=(4,8)),rng.normal(size=(8,2)))]
    spec=QuantSpec(group_size=4);q=tensor(np.eye(4),device);schedule=[(0,1),(1,3)]
    cfg=SolverConfig(backend='torch',device=device,theta_limit=.025,seconds_per_pair=120,wall_seconds=120,
                     checkpoint_path=str(tmp_path/'state.pt'),execution_fingerprint='unit_test_fixture')
    def stop(phase,**details):
        if phase=='solve.pair' and details['pair_index']==1:raise KeyboardInterrupt('fixture interruption')
    with pytest.raises(KeyboardInterrupt):ts.fit_baseline('cellquant',panels,q,spec,schedule,cfg,progress=stop)
    saved=torch.load(cfg.checkpoint_path,map_location=device,weights_only=True)
    assert saved['metadata']['status']=='INTERRUPTED' and saved['metadata']['next_pair']==1
    cfg.resume=True
    resumed=ts.fit_baseline('cellquant',panels,q,spec,schedule,cfg)
    cfg.resume=False;cfg.checkpoint_path=None
    full=ts.fit_baseline('cellquant',panels,q,spec,schedule,cfg)
    torch.testing.assert_close(resumed['q'],full['q'],rtol=1e-10,atol=1e-12)
    cfg.event_limit_per_pair=0;cfg.theta_limit=.7
    limited=ts.fit_baseline('cellquant',panels,q,spec,schedule,cfg)
    assert limited['status']=='RESOURCE_BLOCKED'
    torch.testing.assert_close(limited['q'],q,rtol=0,atol=0)


def test_stable_scale_radius_tie_and_actual_rounding(device):
    from cellquant.events import WorkBudget
    z=tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.],[0.,0.,0.]],device)
    g=ts.GroupOrbit('w',0,0,z,QuantSpec(group_size=4),stable=True)
    sc=g.scale_coeff(0.)
    torch.testing.assert_close(sc,tensor([1/7,0,0],device),rtol=0,atol=0)
    a,ids,_=g.enumerate(-.7,.7,True,WorkBudget(10000,30))
    assert a.device.type==device and ids.device.type==device
    zero=ts.GroupOrbit('w',0,0,torch.zeros_like(z),QuantSpec(group_size=4))
    assert not bool(zero.reconstruction(.1,actual=True)[0].ne(0).any())
    # Fixed scale = 1; -0.5 and +0.5 both use nearest-even zero at the boundary.
    tie=ts.GroupOrbit('w',0,0,tensor([[-.5,0,0],[.5,0,0],[1.5,0,0],[-1.5,0,0]],device),QuantSpec(group_size=4),fixed_scale=tensor(1.,device))
    torch.testing.assert_close(tie.reconstruction(0.,actual=True)[0][:,0],tensor([0,0,2,-2],device),rtol=0,atol=0)


def test_two_row_output_ste_matches_full_objective_updates(device):
    from cellquant.baselines import _ste_proposals as full_ste
    rng=np.random.default_rng(778)
    p=CalibrationPanel('o','output',rng.normal(size=(4,8)),rng.normal(size=(8,3)))
    spec=QuantSpec(group_size=4)
    cpu=PanelState(p,np.eye(4),spec,spec);gpu=ts.PanelState(p,tensor(np.eye(4),device),spec,spec)
    cfg=SolverConfig(theta_limit=.3,device='cpu')
    full=list(full_ste([cpu],(0,1),cfg,6,.02))
    partial=[float(v) for v in ts._ste_proposals([gpu],(0,1),cfg,6,.02)]
    np.testing.assert_allclose(partial,full,rtol=1e-10,atol=1e-12)
