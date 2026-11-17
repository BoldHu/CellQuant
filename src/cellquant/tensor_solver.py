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
                          product_coeff, scale_square, trig_roots, trig_value)


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
        self.bw = risk(dynamic_scales(self.a,self.wspec),self.wg)/self.wnorm
        self.ba = risk(dynamic_scales(self.x.T,self.aspec),self.ag)/self.anorm

    def direct(self, pair, theta):
        if self.panel.placement == 'output':
            rows = list(pair);a = self.a[rows].clone();y = self.target[rows].clone()
            rotate_rows(a,(0,1),theta);rotate_rows(y,(0,1),theta)
            frozen = None if self.fixed_w is None else self.fixed_w[rows]
            ah = torch_quantize(a,self.wspec,scales=frozen)[0]
            error = ah@self.xh-y
            return {'loss':self.loss+(error.square().sum()-self.error[rows].square().sum())/self.denominator,
                    'budget_w':self.bw+(risk(dynamic_scales(a,self.wspec),self.wg)-risk(dynamic_scales(self.a[rows],self.wspec),self.wg))/self.wnorm,
                    'budget_a':self.ba}
        a,x = self.a.clone(),self.x.clone()
        rotate_rows(a.T,pair,theta);rotate_rows(x,pair,theta)
        ah = torch_quantize(a,self.wspec,scales=self.fixed_w)[0]
        xh = torch_quantize(x.T,self.aspec,scales=self.fixed_a)[0].T
        return {'loss':(ah@xh-self.target).square().sum()/self.denominator,
                'budget_w':risk(dynamic_scales(a,self.wspec),self.wg)/self.wnorm,
                'budget_a':risk(dynamic_scales(x.T,self.aspec),self.ag)/self.anorm}

    def accept(self,pair,theta):
        if self.panel.placement == 'input':rotate_rows(self.a.T,pair,theta);rotate_rows(self.x,pair,theta)
        else:rotate_rows(self.a,pair,theta);rotate_rows(self.target,pair,theta)
        self.refresh()


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
                for row,vec in enumerate(vectors):
                    radius = torch.hypot(vec[i],vec[j])
                    for start in sorted({i//g*g,j//g*g}):
                        coeff = vec.new_zeros((g,3));coeff[:,0] = vec[start:start+g]
                        for index,b,c in [(i,vec[i],-vec[j]),(j,vec[j],vec[i])]:
                            if start <= index < start+g:
                                coeff[index-start,0]=0;coeff[index-start,1]=b;coeff[index-start,2]=c
                        changed = coeff[:,1:].ne(0).any(1)
                        if not bool(changed.any()):continue
                        other = coeff[~changed,0].abs();stable = bool(other.numel() and other.max() >= radius)
                        fixed = None if frozen is None else frozen[row,start//g,0]
                        self._add(GroupOrbit(side,row,start,coeff,spec,stable,fixed),scales[row,start//g,0])

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


def solve_pair(states,pair,limits,config):
    if not 0 < config.theta_limit < math.pi:raise ValueError('theta_limit must lie in (0, pi)')
    d = states[0].panel.hidden_size
    if len(pair)!=2 or pair[0]==pair[1] or min(pair)<0 or max(pair)>=d:raise ValueError('Invalid pair')
    synchronize(states[0].a.device);start=time.perf_counter();work=WorkBudget(config.event_limit_per_pair,config.seconds_per_pair)
    orbit=[PanelOrbit(s,pair) for s in states];baseline=aggregate(states);like=states[0].a
    left,right=-config.theta_limit,config.theta_limit
    timings={k:0. for k in ['event_generation','sort','coefficient_updates','roots','boundary_rounding','verification']}
    stats={s:{k:0 for k in ['owner_intervals','member_visits','stable_groups','groups']} for s in ['input','output']}
    groups=[];angles=[like.new_tensor([left,right,0.])];keys=[torch.full((3,),-1,device=like.device,dtype=torch.long)];members=[keys[0]]
    t=time.perf_counter()
    for o in orbit:
        for gid,g in enumerate(o.groups):
            a,ids,st=g.enumerate(left,right,config.screen,work)
            for k,v in st.items():stats[o.state.panel.placement][k]+=v
            angles.append(a);members.append(ids);keys.append(torch.full((len(a),),len(groups),device=like.device,dtype=torch.long));groups.append((o,gid))
    synchronize(like.device);timings['event_generation']=time.perf_counter()-t;t=time.perf_counter()
    angles,inverse=torch.unique(torch.cat(angles),sorted=True,return_inverse=True);keys=torch.cat(keys);members=torch.cat(members)
    synchronize(like.device);timings['sort']=time.perf_counter()-t
    def incident(index,all_groups=False):
        if all_groups:return [(o,g,None) for o,g in groups]
        at=torch.where((inverse==index)&(keys>=0))[0];result=[]
        for key in torch.unique(keys[at]):
            ids=members[at[keys[at]==key]];o,g=groups[int(key)]
            result.append((o,g,None if bool((ids<0).any()) else torch.unique(ids)))
        return result
    def coefficients():return [sum(o.state.panel.objective_weight*getattr(o,k) for o in orbit) for k in ['loss_coeff','bw','ba']]
    def feasible(bw,ba):
        if not config.constrain_budgets:return torch.ones_like(bw,dtype=torch.bool)
        return (bw<=limits['budget_w']+config.budget_tolerance*limits['budget_w'].clamp_min(1)) & (ba<=limits['budget_a']+config.budget_tolerance*limits['budget_a'].clamp_min(1))
    short_theta=like.new_empty(0);short_loss=like.new_empty(0);candidate_count=0;boundary_count=0
    def consider(theta,lc,bw,ba):
        nonlocal short_theta,short_loss,candidate_count
        theta=theta.reshape(-1);candidate_count+=len(theta)
        losses=trig_value(lc,theta);ok=feasible(trig_value(bw,theta),trig_value(ba,theta))&(losses<baseline['loss'])
        short_theta=torch.cat((short_theta,theta[ok]));short_loss=torch.cat((short_loss,losses[ok]))
        if len(short_loss)>32:
            keep=short_loss.topk(32,largest=False).indices;short_theta,short_loss=short_theta[keep],short_loss[keep]
    for n in range(len(angles)-1):
        work.check();lo,hi=angles[n],angles[n+1];mid=lo+(hi-lo)/2;t=time.perf_counter()
        for o,g,ids in incident(n,n==0):o.replace(g,mid,indices=ids)
        synchronize(like.device);timings['coefficient_updates']+=time.perf_counter()-t
        lc,bw,ba=coefficients();t=time.perf_counter();derivative=torch.zeros_like(lc)
        k=torch.arange(1,5,device=like.device,dtype=like.dtype);derivative[1::2]=k*lc[2::2];derivative[2::2]=-k*lc[1::2]
        stationary=trig_roots(derivative,lo,hi);cuts=[torch.stack((lo,hi))]
        if config.constrain_budgets:
            for c,key in [(bw,'budget_w'),(ba,'budget_a')]:
                c=c.clone();c[0]-=limits[key];cuts.append(trig_roots(c,lo,hi))
        cuts=torch.unique(torch.cat(cuts),sorted=True);a,b=cuts[:-1],cuts[1:]
        candidates=torch.cat((stationary,cuts[1:-1],a+(b-a)/2,torch.nextafter(a,b),torch.nextafter(b,a),a+(b-a)*1e-7,b-(b-a)*1e-7))
        consider(candidates,lc,bw,ba);synchronize(like.device);timings['roots']+=time.perf_counter()-t;t=time.perf_counter()
        for idx in ([n,n+1] if n==0 else [n+1]):
            dirty=incident(idx,idx==0 or idx==len(angles)-1);theta=angles[idx]
            for o,g,_ in dirty:o.replace(g,theta,actual=True)
            consider(theta,*coefficients());boundary_count+=1
            for o,g,_ in dirty:o.replace(g,mid)
        synchronize(like.device);timings['boundary_rounding']+=time.perf_counter()-t
    t=time.perf_counter();best={**baseline,'theta':like.new_zeros(())}
    for theta in short_theta[short_loss.argsort()]:
        work.check();measured=[s.direct(pair,theta) for s in states]
        actual={key:sum(s.panel.objective_weight*v[key] for s,v in zip(states,measured)) for key in baseline}
        if bool(feasible(actual['budget_w'],actual['budget_a']) & (actual['loss']<best['loss'])):best={**actual,'theta':theta}
    synchronize(like.device);timings['verification']=time.perf_counter()-t
    return {'status':'COMPLETE','baseline':plain(baseline),'best':plain(best),'accepted':bool(best['theta']!=0),
            'event_count':len(angles),'event_incidences':work.generated,'candidate_count':candidate_count,
            'boundary_rounding_evaluations':boundary_count,'statistics':stats,'incremental_statistics':[o.update_stats for o in orbit],
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


def fit_baseline(method,panels,q0,specs,schedule,config,*,grid_points=129,ste_steps=100,learning_rate=.02,progress=None,budgets=None):
    from .baselines import METHODS
    from .spinquant_adapter import atomic_torch_save
    import hashlib
    if method not in METHODS:raise ValueError(f'Unknown matched method: {method}')
    if getattr(config,'compute_dtype','float64')!='float64':raise ValueError('This numerical protocol requires FP64; no silent precision change')
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
    signature=config_hash({'backend':'torch_resident_v1','method':method,'panels':panel_fingerprint(panels),
        'q0':hashlib.sha256(q.detach().cpu().numpy().tobytes()).hexdigest(),'schedule':schedule,
        'specs':{s.panel.panel_id:[asdict(s.wspec),asdict(s.aspec)] for s in states},'limits':plain(limits),
        'config':{k:v for k,v in asdict(config).items() if k not in {'resume','checkpoint_path'}}})
    log=[];factors=[];offset=0;spent=0.;evaluations=0;status='COMPLETE';reason=None
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
                  'elapsed_seconds':spent+time.perf_counter()-started}
            atomic_torch_save({'q':q.detach(),'metadata':meta},checkpoint)
    save('RUNNING')
    try:
        if method!='hadamard_shared':
            for index in range(offset,len(schedule)):
                synchronize(q.device);remaining=config.wall_seconds-spent-(time.perf_counter()-started)
                if remaining<=0:
                    if cell:raise ResourceLimit('Total solver wall-clock budget exhausted')
                    break
                pair=schedule[index]
                report(progress,'solve.pair',pair_index=index,pair_count=len(schedule),device=str(q.device))
                if cell:
                    from .solver import SolverConfig
                    cfg=SolverConfig(**{**asdict(config),'seconds_per_pair':min(config.seconds_per_pair,remaining)})
                    result=solve_pair(states,pair,limits,cfg)
                    result.update(pair=list(pair),pair_index=index)
                else:
                    end=time.perf_counter()+min(config.seconds_per_pair,remaining)
                    baseline=aggregate(states);best={**baseline,'theta':q.new_zeros(())};count=0
                    if method=='angle_grid_matched':
                        grid=torch.unique(torch.cat((torch.linspace(-config.theta_limit,config.theta_limit,grid_points,device=q.device,dtype=q.dtype),q.new_zeros(1))))
                        proposals=iter(grid[grid.abs().argsort()])
                    else:proposals=_ste_proposals(states,pair,config,ste_steps,learning_rate)
                    for theta in proposals:
                        synchronize(q.device)
                        if time.perf_counter()>=end:break
                        values=[s.direct(pair,theta) for s in states];count+=1;evaluations+=1
                        measured={k:sum(s.panel.objective_weight*v[k] for s,v in zip(states,values)) for k in baseline}
                        feasible=not config.constrain_budgets or all(bool(measured[k]<=limits[k]+config.budget_tolerance*limits[k].clamp_min(1)) for k in limits)
                        if feasible and bool(measured['loss']<best['loss']):best={**measured,'theta':theta}
                    # Explicitly close a suspended STE generator to release its graph.
                    if hasattr(proposals,'close'):proposals.close()
                    result={'pair_index':index,'pair':list(pair),'best':plain(best),'baseline':plain(baseline),
                            'accepted':bool(best['theta']!=0),'evaluations':count}
                if result['accepted']:
                    theta=result['best']['theta']
                    for state in states:state.accept(pair,theta)
                    rotate_rows(q,pair,theta);factors.append({'pair':list(pair),'theta':theta})
                log.append(result);next_pair=index+1;save('RUNNING')
    except ResourceLimit as exc:
        status='RESOURCE_BLOCKED';reason=str(exc);save(status)
    except KeyboardInterrupt:
        # Keep the last atomic completed transaction, never a half-updated Q.
        if checkpoint is not None and cell and checkpoint.is_file():
            safe=torch.load(checkpoint,map_location=q.device,weights_only=True)
            safe['metadata'].update(status='INTERRUPTED',elapsed_seconds=spent+time.perf_counter()-started)
            atomic_torch_save(safe,checkpoint)
        raise
    synchronize(q.device);save(status)
    orth=float(torch.linalg.norm(q.T@q-eye)/math.sqrt(d));synchronize(q.device)
    return {'status':status,'reason':reason,'q':q,'initial':plain(initial),'final':plain(aggregate(states)),
            'limits':plain(limits),'factors':factors,'log':log,'seconds':spent+time.perf_counter()-started,
            'evaluations':evaluations,'completed_pairs':next_pair,'requested_pairs':len(schedule),
            'stopped_by_wall_budget':next_pair<len(schedule) and method!='hadamard_shared','signature':signature,
            'shared_global_r1':True,'orientation':'R1=Q.T','orthogonality_residual':orth,'interval_certified':False,
            'numerical_backend':'torch_resident_v1','numerical_device':str(q.device),'compute_dtype':'float64',
            'root_solver':'torch_complex_shifted_qr_v1',
            'host_work':'Python scheduling/scalar decisions and checkpoint/hash I/O; no NumPy/SciPy solver arithmetic'}
