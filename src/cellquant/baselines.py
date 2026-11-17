"""Same-panel, same-coordinate-budget controls; no base-model training."""
from dataclasses import asdict
import math
import time
import numpy as np

from .quantization import QuantSpec, torch_quantize
from .progress import report
from .solver import (SolverConfig, PanelState, aggregate, rotate_rows,
                     fit_shared_rotation,panel_specs)


METHODS=('hadamard_shared','cellquant','angle_grid_matched','givens_ste_matched','fixed_scale_search_matched')


def fit_baseline(method,panels,q0,specs,schedule,config=None,*,grid_points=129,ste_steps=100,learning_rate=.02,progress=None):
    config=config or SolverConfig()
    if config.backend=='torch':
        from .tensor_solver import fit_baseline as fit_tensor
        return fit_tensor(method,panels,q0,specs,schedule,config,grid_points=grid_points,ste_steps=ste_steps,learning_rate=learning_rate,progress=progress)
    if config.backend!='numpy_reference':raise ValueError('Unknown solver backend')
    if method not in METHODS:raise ValueError(f"Unknown matched method: {method}")
    if method in {'cellquant','fixed_scale_search_matched'}:
        cfg=SolverConfig(**{**asdict(config),'fixed_scales':method=='fixed_scale_search_matched'})
        return fit_shared_rotation(panels,q0,specs,schedule,config=cfg,progress=progress)
    if config.resume:raise ValueError("Grid/STE resume requires a completed method artifact; partial optimizer resume is not supported")
    start=time.perf_counter();q=np.asarray(q0,dtype=np.float64).copy()
    states=[]
    for p in panels:
        report(progress,'solve.initialize_panel',panel_id=p.panel_id,device=config.device)
        states.append(PanelState(p,q,*panel_specs(specs,p),device=config.device))
    initial=aggregate(states);limits={k:initial[k] for k in ('budget_w','budget_a')}
    factors=[];log=[];evaluations=0
    def evaluate(pair,theta):
        nonlocal evaluations
        values=[s.direct(pair,theta) for s in states];evaluations+=1
        return {k:sum(s.panel.objective_weight*v[k] for s,v in zip(states,values)) for k in initial}
    if method!='hadamard_shared':
        for index,pair in enumerate(schedule):
            remaining=config.wall_seconds-(time.perf_counter()-start)
            if remaining<=0:break
            report(progress,'solve.pair',pair_index=index,pair_count=len(schedule),device=config.device)
            stop=time.perf_counter()+min(config.seconds_per_pair,remaining)
            baseline=aggregate(states);best=dict(baseline,theta=0.)
            if method=='angle_grid_matched':
                proposals=iter(sorted(set(np.linspace(-config.theta_limit,config.theta_limit,grid_points).tolist()+[0.]),key=abs))
            else:
                proposals=_ste_proposals(states,pair,config,ste_steps,learning_rate)
            count=0
            for theta in proposals:
                if time.perf_counter()>=stop:break
                v=evaluate(pair,float(theta));count+=1
                feasible=not config.constrain_budgets or all(v[k]<=limits[k]+config.budget_tolerance*max(1.,limits[k]) for k in limits)
                if feasible and v['loss']<best['loss']:best=dict(v,theta=float(theta))
            accepted=best['theta']!=0.
            if accepted:
                for state in states:state.accept(pair,best['theta'])
                rotate_rows(q,pair,best['theta']);factors.append({'pair':list(pair),'theta':best['theta']})
            log.append({'pair_index':index,'pair':list(pair),'best':best,'baseline':baseline,'accepted':accepted,'evaluations':count})
    return {'status':'COMPLETE','q':q,'initial':initial,'final':aggregate(states),'limits':limits,
            'factors':factors,'log':log,'seconds':time.perf_counter()-start,'evaluations':evaluations,
            'completed_pairs':len(log),'requested_pairs':len(schedule),
            'stopped_by_wall_budget':len(log)<len(schedule) and method!='hadamard_shared',
            'shared_global_r1':True,'orientation':'R1=Q.T',
            'orthogonality_residual':float(np.linalg.norm(q.T@q-np.eye(len(q)))/math.sqrt(len(q)))}


def _ste_proposals(states,pair,config,steps,lr):
    import torch
    device=config.device
    theta=torch.zeros((),dtype=torch.float64,device=device,requires_grad=True)
    optimizer=torch.optim.AdamW([theta],lr=lr,weight_decay=0.)
    tensors=[(torch.tensor(s.a,device=device),torch.tensor(s.x,device=device),torch.tensor(s.target,device=device)) for s in states]
    i,j=pair
    def rotated(z,t):
        rows=z[[i,j]]
        vals=torch.stack((t.cos()*rows[0]-t.sin()*rows[1],t.sin()*rows[0]+t.cos()*rows[1]))
        return z.index_copy(0,torch.tensor([i,j],device=device),vals)
    yield 0.
    for _ in range(steps):
        optimizer.zero_grad();loss=theta*0
        for state,(a,x,y) in zip(states,tensors):
            if state.panel.placement=='input':
                ar=rotated(a.T,theta).T;xr=rotated(x,theta);yr=y
            else: ar=rotated(a,theta);xr=x;yr=rotated(y,theta)
            ah=torch_quantize(ar,state.wspec,ste=True)[0]
            xh=torch_quantize(xr.T,state.aspec,ste=True)[0].T
            loss=loss+state.panel.objective_weight*(ah@xh-yr).square().sum()/state.denominator
        loss.backward();optimizer.step()
        with torch.no_grad():theta.clamp_(-config.theta_limit,config.theta_limit)
        if device!='cpu':torch.cuda.synchronize()
        yield float(theta.detach().cpu())
