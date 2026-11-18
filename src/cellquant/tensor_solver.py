"""GPU-resident PyTorch implementation of the shared-Q RTN panel solver.

CPU/NumPy reference lives in solver.py. This implementation keeps panel math,
events, sorting, polynomial roots, finalist checks and matched controls on the
selected torch device. Host work is scheduling, scalar branching and artifact I/O.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

from .events import WorkBudget, ResourceLimit
from .progress import report
from .quantization import torch_quantize
from .runtime import config_hash
from .solver import panel_specs, panel_fingerprint
from .tensor_math import (scalar, basis, affine_roots, owner_intervals, square_coeff,
                          product_coeff, scale_square)

CELL_ENGINE_VERSION = 'torch_batched_bilateral_prefix_v3'
MATCHED_GRID_POINTS = 129


def synchronize(device):
    if torch.device(device).type == 'cuda':torch.cuda.synchronize(device)


def numerical_context(device):
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; CPU fallback is disabled')
    torch.backends.cuda.matmul.allow_tf32 = False


def hadamard_rotation(d, seed, device='cuda'):
    from .hadamard import torch_hadamard
    generator = torch.Generator(device=device).manual_seed(seed)
    signs = torch.randint(0, 2, (d,), device=device, generator=generator)*2-1
    return torch_hadamard(torch.eye(d, dtype=torch.float64, device=device), 1/math.sqrt(d))*signs


def rotate_rows(z, pair, theta):
    i,j = pair;t = scalar(theta, z);u,v = z[i].clone(), z[j].clone()
    z[i],z[j] = t.cos()*u-t.sin()*v, t.sin()*u+t.cos()*v


def dynamic_scales(z, spec):
    return z.reshape(z.shape[0], -1, spec.width(z.shape[1])).abs().amax(-1, keepdim=True)/spec.qmax


def risk(scales, width):return scales.square().sum()*width/4


class PanelState:
    def __init__(self, panel, q, wspec, aspec, *, fixed=False):
        self.panel,self.wspec,self.aspec = panel,wspec,aspec
        self.wg,self.ag = wspec.width(panel.weight.shape[1]),aspec.width(panel.activation.shape[0])
        w = torch.as_tensor(panel.weight, dtype=torch.float64, device=q.device)
        x = torch.as_tensor(panel.activation, dtype=torch.float64, device=q.device)
        if not bool(torch.isfinite(w).all() & torch.isfinite(x).all()):raise ValueError('Nonfinite panel')
        self.a = w@q.T if panel.placement == 'input' else q@w
        self.x = q@x if panel.placement == 'input' else x.clone()
        self.target = w@x
        if panel.placement == 'output':self.target = q@self.target
        self.denominator = self.target.square().sum().clamp_min(1e-30)
        self.wnorm,self.anorm = w.square().sum().clamp_min(1e-30),x.square().sum().clamp_min(1e-30)
        self.fixed_w = dynamic_scales(self.a, wspec) if fixed else None
        self.fixed_a = dynamic_scales(self.x.T, aspec) if fixed else None
        self.refresh()

    def refresh(self):
        self.ah,_,self.sw = torch_quantize(self.a, self.wspec, scales=self.fixed_w)
        xh,_,self.sx = torch_quantize(self.x.T, self.aspec, scales=self.fixed_a)
        self.xh = xh.T;self.error = self.ah@self.xh-self.target
        self.loss = self.error.square().sum()/self.denominator
        self.physical_sw = self.sw if self.fixed_w is None else dynamic_scales(self.a,self.wspec)
        physical_sx = self.sx if self.fixed_a is None else dynamic_scales(self.x.T,self.aspec)
        self.bw = risk(self.physical_sw,self.wg)/self.wnorm
        self.ba = risk(physical_sx,self.ag)/self.anorm

    def _output_rows(self, rows, theta):
        a, target = self.a[rows], self.target[rows]
        rotate_rows(a, (0, 1), theta)
        rotate_rows(target, (0, 1), theta)
        frozen = None if self.fixed_w is None else self.fixed_w[rows]
        ah, _, scales = torch_quantize(a, self.wspec, scales=frozen)
        physical = scales if frozen is None else dynamic_scales(a, self.wspec)
        return a, target, ah, physical, ah @ self.xh - target

    def direct(self, pair, theta):
        if self.panel.placement == 'output':
            rows = list(pair)
            _, _, _, physical, error = self._output_rows(rows, theta)
            return {'loss':self.loss+(error.square().sum()-self.error[rows].square().sum())/self.denominator,
                    'budget_w':self.bw+(risk(physical,self.wg)-risk(self.physical_sw[rows],self.wg))/self.wnorm,
                    'budget_a':self.ba}
        a,x = self.a.clone(),self.x.clone()
        rotate_rows(a.T,pair,theta);rotate_rows(x,pair,theta)
        ah = torch_quantize(a,self.wspec,scales=self.fixed_w)[0]
        xh = torch_quantize(x.T,self.aspec,scales=self.fixed_a)[0].T
        return {'loss':(ah@xh-self.target).square().sum()/self.denominator,
                'budget_w':risk(dynamic_scales(a,self.wspec),self.wg)/self.wnorm,
                'budget_a':risk(dynamic_scales(x.T,self.aspec),self.ag)/self.anorm}

    def accept(self,pair,theta):
        if self.panel.placement == 'input':
            rotate_rows(self.a.T,pair,theta)
            rotate_rows(self.x,pair,theta)
            self.refresh()
            return
        # Left rotation changes two complete weight/target rows. All other
        # rows and the quantized activation are already current.
        rows = list(pair)
        a, target, ah, physical, error = self._output_rows(rows, theta)
        self.a[rows], self.target[rows], self.ah[rows] = a, target, ah
        self.physical_sw[rows], self.error[rows] = physical, error
        self.bw = risk(self.physical_sw, self.wg) / self.wnorm
        self.loss = self.error.square().sum() / self.denominator


class GroupOrbit:
    def __init__(self,side,row,start,coeff,spec,stable=False,fixed_scale=None):
        self.side,self.row,self.start,self.coeff,self.spec = side,row,start,coeff,spec
        self.stable,self.fixed_scale = stable,fixed_scale

    def scale_coeff(self,theta,dynamic=False):
        if self.fixed_scale is not None and not dynamic:
            return torch.stack((self.fixed_scale,self.coeff.new_zeros(()),self.coeff.new_zeros(())))
        if self.stable:
            fixed = self.coeff[:,1:].eq(0).all(1)
            result = self.coeff.new_zeros(3);result[0] = self.coeff[fixed,0].abs().max()/self.spec.qmax
            return result
        v = self.coeff@basis(theta,self.coeff);i = v.abs().argmax()
        return v[i].sign()*self.coeff[i]/self.spec.qmax

    def reconstruction(self,theta,*,actual=False,indices=None):
        sc = self.scale_coeff(theta);b = basis(theta,self.coeff);scale = (sc@b).clamp_min(0)
        coeff = self.coeff if indices is None else self.coeff[indices]
        code = ((coeff@b)/torch.where(scale>0,scale,torch.ones_like(scale))).round().clamp(self.spec.qmin,self.spec.qmax)
        if actual:
            result = coeff.new_zeros((len(coeff),3));result[:,0] = code*scale
            return result,sc
        return code[:,None]*sc,sc

    def enumerate(self,left,right,screen,budget):
        moving = torch.where(self.coeff[:,1:].ne(0).any(1))[0]
        intervals = owner_intervals(self.coeff,left,right,self.stable and screen,budget)
        roots=[];members=[];visits=0
        for lo,hi,owner,sign in intervals:
            budget.check();roots.append(torch.stack((lo,hi)))
            members.append(torch.full((2,),-1,dtype=torch.long,device=self.coeff.device))
            scale = self.scale_coeff((lo+hi)/2) if self.fixed_scale is not None or self.stable else sign*self.coeff[owner]/self.spec.qmax
            ids = moving if (self.stable and screen) or self.fixed_scale is not None else torch.arange(len(self.coeff),device=self.coeff.device)
            visits += len(ids)
            thresholds = torch.arange(self.spec.qmin+.5,self.spec.qmax,device=self.coeff.device,dtype=self.coeff.dtype)
            z = self.coeff[ids][None,:,:]-thresholds[:,None,None]*scale
            t,ix = affine_roots(z,lo,hi);budget.check(len(t))
            if t.numel():roots.append(t);members.append(ids[ix % len(ids)])
        return torch.cat(roots),torch.cat(members),{'owner_intervals':len(intervals),'member_visits':visits,'stable_groups':int(self.stable),'groups':1}


class PanelOrbit:
    def __init__(self,state,pair):
        self.state,self.pair = state,pair;output = state.panel.placement == 'output';i,j = pair
        a = state.a[list(pair)] if output else state.a
        ah = state.ah[list(pair)] if output else state.ah
        self.aw = a.new_zeros((3,*a.shape));self.ax = a.new_zeros((3,*state.x.shape))
        self.aw[0],self.ax[0] = ah,state.xh
        self.error = a.new_zeros((5,a.shape[0],state.x.shape[1]))
        self.error[0] = state.error[list(pair)] if output else state.error
        remainder = state.loss-self.error[0].square().sum()/state.denominator if output else a.new_zeros(())
        if output:
            target = state.target[list(pair)]
            self.error[0] += target;self.error[1] -= target
            self.error[2,0] += target[1];self.error[2,1] -= target[0]
        self.loss_coeff = square_coeff(self.error)/state.denominator;self.loss_coeff[0] += remainder
        self.bw,self.ba = a.new_zeros(5),a.new_zeros(5);self.bw[0],self.ba[0] = state.bw,state.ba
        self.groups=[];self.group_scales=[]
        self.update_stats={'code_member_updates':0,'owner_member_updates':0,'row_column_updates':0,'full_gemm_per_event':0}
        if output:
            for row in range(2):
                coeff = a.new_zeros((a.shape[1],3));coeff[:,1] = a[row];coeff[:,2] = -a[1] if row==0 else a[0]
                for start in range(0,a.shape[1],state.wg):
                    g = start//state.wg;frozen = None if state.fixed_w is None else state.fixed_w[pair[row],g,0]
                    self._add(GroupOrbit('w',row,start,coeff[start:start+state.wg].clone(),state.wspec,False,frozen),state.sw[pair[row],g,0])
        else:
            for side,vectors,spec,g,scales,frozen in [('w',state.a,state.wspec,state.wg,state.sw,state.fixed_w),('a',state.x.T,state.aspec,state.ag,state.sx,state.fixed_a)]:
                u, v = vectors[:, i], vectors[:, j]
                radius = torch.hypot(u, v)
                active = (u != 0) | (v != 0)
                starts = sorted({i//g*g, j//g*g})
                blocks, stable, initial = [], [], []
                for start in starts:
                    coeff = vectors.new_zeros((len(vectors), g, 3))
                    coeff[:, :, 0] = vectors[:, start:start+g]
                    for index, b, c in [(i, u, -v), (j, v, u)]:
                        if start <= index < start+g:
                            col = index-start
                            coeff[:, col, 0] = 0
                            coeff[:, col, 1], coeff[:, col, 2] = b, c
                    blocks.append(coeff)
                    stable.append(coeff[:, :, 0].abs().amax(1) >= radius)
                    sc = vectors.new_zeros((len(vectors), 3))
                    sc[:, 0] = scales[:, start//g, 0]
                    initial.append(sc)
                # One host transfer for scheduling flags across all rows.
                flags = torch.stack([active, *stable], dim=1).tolist()
                for row, (changed, *stable_flags) in enumerate(flags):
                    if not changed:continue
                    for block, start in enumerate(starts):
                        fixed = None if frozen is None else frozen[row,start//g,0]
                        self.groups.append(GroupOrbit(side,row,start,blocks[block][row],spec,stable_flags[block],fixed))
                        self.group_scales.append(initial[block][row])

    def _add(self,group,scale):
        self.groups.append(group);sc = group.coeff.new_zeros(3);sc[0] = scale;self.group_scales.append(sc)

    def replace(self,gid,theta,*,actual=False,indices=None):
        group = self.groups[gid];old_scale = self.group_scales[gid];new_scale = group.scale_coeff(theta)
        changed_owner = not torch.equal(new_scale,old_scale)
        ix = torch.arange(len(group.coeff),device=group.coeff.device) if changed_owner or indices is None else indices
        if not ix.numel():return
        cols = group.start+ix
        dest = self.aw[:,group.row,cols].clone() if group.side=='w' else self.ax[:,cols,group.row].clone()
        replacement = group.reconstruction(theta,actual=actual,indices=ix)[0].T
        delta = replacement-dest;active = delta.ne(0).any(0);changed_cols = cols[active]
        if bool(active.any()):
            if group.side=='w':
                old_error = self.error[:,group.row:group.row+1].clone()
                self.error[:,group.row:group.row+1] += product_coeff(delta[:,None,active],self.ax[:,changed_cols,:])
                self.aw[:,group.row,changed_cols] = replacement[:,active]
                new_error = self.error[:,group.row:group.row+1]
            else:
                old_error = self.error[:,:,group.row:group.row+1].clone()
                self.error[:,:,group.row:group.row+1] += product_coeff(self.aw[:,:,changed_cols],delta[:,active,None])
                self.ax[:,changed_cols,group.row] = replacement[:,active]
                new_error = self.error[:,:,group.row:group.row+1]
            self.loss_coeff += (square_coeff(new_error)-square_coeff(old_error))/self.state.denominator
            self.update_stats['row_column_updates'] += 1
            self.update_stats['owner_member_updates' if changed_owner else 'code_member_updates'] += int(active.sum())
        sc = group.scale_coeff(theta,dynamic=True);old_risk = getattr(group,'_risk_scale',old_scale)
        value,norm = (self.bw,self.state.wnorm) if group.side=='w' else (self.ba,self.state.anorm)
        value += (scale_square(sc,len(group.coeff))-scale_square(old_risk,len(group.coeff)))/norm
        group._risk_scale = sc.clone();self.group_scales[gid] = new_scale.clone()


def aggregate(states):
    return {k:sum(s.panel.objective_weight*getattr(s,a) for s in states) for k,a in [('loss','loss'),('budget_w','bw'),('budget_a','ba')]}


def plain(values):
    return {k:float(v) if torch.is_tensor(v) else v for k,v in values.items()}


class EventIndex:
    """GPU incidence index: sort once, then visit only the current boundary.

    Re-scanning the entire incidence vector at every cell is quadratic in the
    number of events. Offsets are host scalar control; event arrays stay on GPU.
    """
    def __init__(self,inverse,keys,members,event_count):
        valid=keys>=0
        order=torch.argsort(inverse[valid],stable=True)
        self.keys,self.members=keys[valid][order],members[valid][order]
        counts=torch.bincount(inverse[valid],minlength=event_count)
        self.offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)))

    def at(self,index):
        start,end=int(self.offsets[index]),int(self.offsets[index+1])
        return self.keys[start:end],self.members[start:end]


def solve_pair(states,pair,limits,config,*,progress=None):
    trace={'pair':list(pair),'phase':'orbit_initialization','groups_enumerated':0,
           'event_incidences':0,'event_count':None,'cells_completed':0,'cell_count':None,
           'started':time.perf_counter(),'last_emitted':0.}
    def snapshot():
        result={k:v for k,v in trace.items() if k not in {'started','last_emitted','work'}}
        result['seconds']=time.perf_counter()-trace['started']
        if 'work' in trace:result['event_incidences']=trace['work'].generated
        return result
    def emit(phase,*,force=False,**details):
        trace.update(phase=phase,**details)
        now=time.perf_counter()
        if force or now-trace['last_emitted']>=30.:
            trace['last_emitted']=now
            details=snapshot();details['search_phase']=details.pop('phase')
            report(progress,'solve.pair.phase',**details)
    try:
        return _solve_pair(states,pair,limits,config,trace,emit)
    except ResourceLimit as exc:
        exc.diagnostics=snapshot()
        details=dict(exc.diagnostics);details['search_phase']=details.pop('phase')
        report(progress,'solve.pair.blocked',reason=str(exc),**details)
        raise


def _solve_pair(states,pair,limits,config,trace,emit):
    if config.cell_batch_size<1:raise ValueError('cell_batch_size must be positive')
    if not 0 < config.theta_limit < math.pi:raise ValueError('theta_limit must lie in (0, pi)')
    d = states[0].panel.hidden_size
    if len(pair)!=2 or pair[0]==pair[1] or min(pair)<0 or max(pair)>=d:raise ValueError('Invalid pair')
    synchronize(states[0].a.device);start=time.perf_counter();work=WorkBudget(config.event_limit_per_pair,config.seconds_per_pair)
    trace['work']=work
    emit('orbit_initialization',force=True)
    orbit=[PanelOrbit(s,pair) for s in states];baseline=aggregate(states);like=states[0].a
    left,right=-config.theta_limit,config.theta_limit
    timings={k:0. for k in ['event_generation','sort','cell_coefficients_and_boundaries','candidate_search','verification']}
    trace['timings']=timings
    stats={s:{k:0 for k in ['owner_intervals','member_visits','stable_groups','groups']} for s in ['input','output']}
    angles=[like.new_tensor([left,right,0.])];orbit_events=[]
    t=time.perf_counter()
    for o in orbit:
        local_events=[];orbit_events.append(local_events)
        emit('event_generation',force=True,panel_id=o.state.panel.panel_id)
        batched = {}
        if config.screen and o.state.panel.placement == 'input':
            from .tensor_input_cells import stable_group_events
            batched = stable_group_events(o, left, right, work)
        for gid,g in enumerate(o.groups):
            emit('event_generation',group_id=gid,group_count=len(o.groups))
            a,ids,st=batched[gid] if gid in batched else g.enumerate(left,right,config.screen,work)
            for k,v in st.items():stats[o.state.panel.placement][k]+=v
            angles.append(a);local_events.append((a,ids))
            trace['groups_enumerated']+=1
    synchronize(like.device);timings['event_generation']=time.perf_counter()-t;t=time.perf_counter()
    emit('event_sort',force=True,panel_id=None,group_id=None,group_count=None)
    angles=torch.unique(torch.cat(angles),sorted=True)
    synchronize(like.device);timings['sort']=time.perf_counter()-t
    trace.update(event_count=len(angles),cell_count=len(angles)-1)
    from .tensor_cells import output_terms,input_term,search_terms,constant_term
    terms=[];t=time.perf_counter()
    for o,events in zip(orbit,orbit_events):
        work.check()
        if not events:terms.append(constant_term(o.state,angles[[0,-1]]))
        elif o.state.panel.placement=='output':terms.extend(output_terms(o,events,work,config.cell_batch_size,emit))
        else:terms.append(input_term(o,events,work,emit,config.cell_batch_size))
    synchronize(like.device);timings['cell_coefficients_and_boundaries']=time.perf_counter()-t
    t=time.perf_counter()
    short_theta,short_loss,search_stats=search_terms(terms,angles,baseline,limits,config,work,trace,emit,config.cell_batch_size)
    synchronize(like.device);timings['candidate_search']=time.perf_counter()-t
    def feasible(bw,ba):
        if not config.constrain_budgets:return torch.ones_like(bw,dtype=torch.bool)
        return (bw<=limits['budget_w']+config.budget_tolerance*limits['budget_w'].clamp_min(1)) & (ba<=limits['budget_a']+config.budget_tolerance*limits['budget_a'].clamp_min(1))
    t=time.perf_counter();best={**baseline,'theta':like.new_zeros(())}
    emit('verification',force=True)
    for theta in short_theta[short_loss.argsort()]:
        work.check();measured=[s.direct(pair,theta) for s in states]
        actual={key:sum(s.panel.objective_weight*v[key] for s,v in zip(states,measured)) for key in baseline}
        if bool(feasible(actual['budget_w'],actual['budget_a']) & (actual['loss']<best['loss'])):best={**actual,'theta':theta}
    synchronize(like.device);timings['verification']=time.perf_counter()-t
    work.check()
    return {'status':'COMPLETE','baseline':plain(baseline),'best':plain(best),'accepted':bool(best['theta']!=0),
            'event_count':len(angles),'event_incidences':work.generated,**search_stats,
            'cell_engine':CELL_ENGINE_VERSION,'statistics':stats,'incremental_statistics':[o.update_stats for o in orbit],
            'timings':timings,'seconds':time.perf_counter()-start,'interval_certified':False,
            'candidate_policy':'32 lowest predicted feasible candidates; actual full-group verification',
            'scope':'Torch FP64 numerical cell search; not certified/global optimal','numerical_device':str(like.device)}


def _ste_proposals(states,pair,config,steps,lr):
    theta=torch.zeros((),dtype=torch.float64,device=states[0].a.device,requires_grad=True)
    optimizer=torch.optim.AdamW([theta],lr=lr,weight_decay=0.)
    indices=torch.tensor(pair,device=theta.device)
    def pair_rows(z):
        u,v=z[indices]
        return torch.stack((theta.cos()*u-theta.sin()*v,theta.sin()*u+theta.cos()*v))
    yield theta.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad();loss=theta*0
        for state in states:
            if state.panel.placement=='output':
                # Row-wise groups: only the two changed output rows have a
                # theta derivative. This is the exact full objective gradient.
                ar=pair_rows(state.a);yr=pair_rows(state.target)
                ah=torch_quantize(ar,state.wspec,ste=True)[0]
                value=state.loss+((ah@state.xh-yr).square().sum()-state.error[indices].square().sum())/state.denominator
            else:
                ar=state.a.T.index_copy(0,indices,pair_rows(state.a.T)).T
                xr=state.x.index_copy(0,indices,pair_rows(state.x))
                ah=torch_quantize(ar,state.wspec,ste=True)[0]
                xh=torch_quantize(xr.T,state.aspec,ste=True)[0].T
                value=(ah@xh-state.target).square().sum()/state.denominator
            loss=loss+state.panel.objective_weight*value
        loss.backward();optimizer.step()
        with torch.no_grad():theta.clamp_(-config.theta_limit,config.theta_limit)
        yield theta.detach().clone()


def grid_angles(like, theta_limit, count=MATCHED_GRID_POINTS):
    grid = torch.linspace(-theta_limit, theta_limit, count, device=like.device, dtype=like.dtype)
    grid = torch.unique(torch.cat((grid, like.new_zeros(1))))
    return grid[grid.abs().argsort()]


def evaluate_proposals(states, pair, limits, config, proposals, deadline):
    baseline = aggregate(states)
    best = {**baseline, 'theta': states[0].a.new_zeros(())}
    count = 0
    try:
        for theta in proposals:
            synchronize(states[0].a.device)
            if time.perf_counter() >= deadline:
                break
            values = [state.direct(pair, theta) for state in states]
            measured = {key: sum(s.panel.objective_weight * v[key] for s, v in zip(states, values))
                        for key in baseline}
            count += 1
            feasible = not config.constrain_budgets or all(
                bool(measured[k] <= cap + config.budget_tolerance * cap.clamp_min(1))
                for k, cap in limits.items())
            if feasible and bool(measured['loss'] < best['loss']):
                best = {**measured, 'theta': theta}
    finally:
        if hasattr(proposals, 'close'):
            proposals.close()
    if not count:
        raise ResourceLimit('Pair wall-clock budget exhausted before evaluating a proposal')
    return {'pair': list(pair), 'best': plain(best), 'baseline': plain(baseline),
            'accepted': bool(best['theta'] != 0), 'evaluations': count}


def fit_baseline(method,panels,q0,specs,schedule,config,*,grid_points=MATCHED_GRID_POINTS,ste_steps=100,learning_rate=.02,progress=None,budgets=None):
    from .baselines import METHODS
    from .spinquant_adapter import atomic_torch_save
    import hashlib
    if method not in METHODS:raise ValueError(f'Unknown matched method: {method}')
    if getattr(config,'compute_dtype','float64')!='float64':raise ValueError('This numerical protocol requires FP64; no silent precision change')
    if config.cell_batch_size<1:raise ValueError('cell_batch_size must be positive')
    numerical_context(config.device);synchronize(config.device);started=time.perf_counter()
    q=torch.as_tensor(q0,dtype=torch.float64,device=config.device).clone();d=len(q)
    panels=list(panels);schedule=[tuple(map(int,p)) for p in schedule]
    if not panels or len({p.panel_id for p in panels})!=len(panels):raise ValueError('Nonempty unique panels required')
    if q.shape!=(d,d) or any(p.hidden_size!=d for p in panels):raise ValueError('Shared full hidden dimension required')
    eye=torch.eye(d,device=q.device,dtype=q.dtype)
    if not bool(torch.isfinite(q).all()) or bool(torch.linalg.norm(q.T@q-eye)/math.sqrt(d)>1e-6):raise ValueError('Nonorthogonal Q')
    states=[]
    for panel in panels:
        report(progress,'solve.initialize_panel',panel_id=panel.panel_id,device=str(q.device))
        states.append(PanelState(panel,q,*panel_specs(specs,panel),fixed=method=='fixed_scale_search_matched'))
    initial=aggregate(states)
    limits={k:initial[k] if budgets is None else scalar(budgets[k],q) for k in ['budget_w','budget_a']}
    cell=method in {'cellquant','fixed_scale_search_matched'}
    if config.resume and not cell:raise ValueError('Partial grid/STE resume is unsupported')
    # CPU copies here are explicit checksum/serialization work, never solver arithmetic.
    signature=config_hash({'backend':CELL_ENGINE_VERSION,'method':method,'panels':panel_fingerprint(panels),
        'q0':hashlib.sha256(q.detach().cpu().numpy().tobytes()).hexdigest(),'schedule':schedule,
        'specs':{s.panel.panel_id:[asdict(s.wspec),asdict(s.aspec)] for s in states},'limits':plain(limits),
        'config':{k:v for k,v in asdict(config).items() if k not in {'resume','checkpoint_path'}}})
    log=[];factors=[];offset=0;spent=0.;evaluations=0;status='COMPLETE';reason=None;resource_diagnostics=None
    checkpoint=Path(config.checkpoint_path) if config.checkpoint_path else None
    if config.resume:
        if checkpoint is None:raise ValueError('Resume requires a checkpoint')
        saved=torch.load(checkpoint,map_location=q.device,weights_only=True);meta=saved['metadata']
        if meta['signature']!=signature:raise ValueError('Checkpoint provenance changed; create a new run')
        q=saved['q'];log=meta['log'];factors=meta['factors'];offset=meta['next_pair'];spent=meta['elapsed_seconds']
        for factor in factors:
            for state in states:state.accept(tuple(factor['pair']),factor['theta'])
    next_pair=offset
    def save(state):
        if checkpoint is not None and cell:
            synchronize(q.device)
            meta={'signature':signature,'status':state,'log':log,'factors':factors,'next_pair':next_pair,
                  'elapsed_seconds':spent+time.perf_counter()-started,'reason':reason,'resource_diagnostics':resource_diagnostics}
            atomic_torch_save({'q':q.detach(),'metadata':meta},checkpoint)
    save('RUNNING')
    synchronize(q.device)
    loop_started = time.perf_counter()
    try:
        if method!='hadamard_shared':
            for index in range(offset,len(schedule)):
                synchronize(q.device);remaining=config.wall_seconds-spent-(time.perf_counter()-started)
                if remaining<=0:
                    raise ResourceLimit('Total solver wall-clock budget exhausted')
                pair=schedule[index]
                report(progress,'solve.pair',pair_index=index,pair_count=len(schedule),device=str(q.device))
                if cell:
                    from .solver import SolverConfig
                    cfg=SolverConfig(**{**asdict(config),'seconds_per_pair':min(config.seconds_per_pair,remaining)})
                    result=solve_pair(states,pair,limits,cfg,progress=lambda phase,**d:report(progress,phase,pair_index=index,**d))
                    result.update(pair=list(pair),pair_index=index)
                else:
                    end=time.perf_counter()+min(config.seconds_per_pair,remaining)
                    if method=='angle_grid_matched':
                        proposals=grid_angles(q,config.theta_limit,grid_points)
                    else:proposals=_ste_proposals(states,pair,config,ste_steps,learning_rate)
                    result=evaluate_proposals(states,pair,limits,config,proposals,end)
                    result['pair_index']=index
                    evaluations+=result['evaluations']
                if result['accepted']:
                    theta=result['best']['theta']
                    for state in states:state.accept(pair,theta)
                    rotate_rows(q,pair,theta);factors.append({'pair':list(pair),'theta':theta})
                log.append(result);next_pair=index+1;save('RUNNING')
    except ResourceLimit as exc:
        status='RESOURCE_BLOCKED';reason=str(exc)
        resource_diagnostics=getattr(exc,'diagnostics',{'phase':'method_wall_budget','completed_pairs':next_pair})
        save(status)
    except KeyboardInterrupt:
        # Keep the last atomic completed transaction, never a half-updated Q.
        if checkpoint is not None and cell and checkpoint.is_file():
            safe=torch.load(checkpoint,map_location=q.device,weights_only=True)
            safe['metadata'].update(status='INTERRUPTED',elapsed_seconds=spent+time.perf_counter()-started)
            atomic_torch_save(safe,checkpoint)
        raise
    synchronize(q.device)
    loop_seconds = time.perf_counter() - loop_started
    orth=float(torch.linalg.norm(q.T@q-eye)/math.sqrt(d));synchronize(q.device)
    save(status)
    elapsed = spent + time.perf_counter() - started
    if status == 'COMPLETE' and method != 'hadamard_shared' and elapsed > config.wall_seconds:
        status, reason = 'RESOURCE_BLOCKED', 'Total solver wall-clock budget exhausted'
        resource_diagnostics = {'phase': 'method_wall_budget', 'completed_pairs': next_pair}
        save(status)
    return {'status':status,'reason':reason,'resource_diagnostics':resource_diagnostics,'q':q,'initial':plain(initial),'final':plain(aggregate(states)),
            'limits':plain(limits),'factors':factors,'log':log,'seconds':elapsed,
            'this_invocation_pair_loop_seconds':loop_seconds,
            'evaluations':evaluations,'completed_pairs':next_pair,'requested_pairs':len(schedule),
            'stopped_by_wall_budget':next_pair<len(schedule) and method!='hadamard_shared','signature':signature,
            'shared_global_r1':True,'orientation':'R1=Q.T','orthogonality_residual':orth,'interval_certified':False,
            'numerical_backend':CELL_ENGINE_VERSION,'numerical_device':str(q.device),'compute_dtype':'float64',
            'root_solver':'torch_complex_shifted_qr_v1',
            'host_work':'Python scheduling/scalar decisions and checkpoint/hash I/O; no NumPy/SciPy solver arithmetic'}
