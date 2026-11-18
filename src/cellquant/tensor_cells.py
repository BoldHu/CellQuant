"""Batched piecewise objectives on the tensor device.

Output groups use code-jump prefix sums, not a dense GEMM per angle. Input
panels retain the bilateral incremental oracle, evaluated at their own events.
All terms are then merged into one shared-Q objective before angle selection.
Host lists below contain only scheduling indices, never floating point data.
"""
from dataclasses import dataclass
import torch

from .tensor_math import basis, scale_square, trig_value, trig_roots


@dataclass
class CellTerm:
    angles: torch.Tensor
    loss: torch.Tensor
    bw: torch.Tensor
    ba: torch.Tensor
    boundary: torch.Tensor  # actual loss, weight risk, activation risk
    weight: float


def constant_term(state,angles):
    loss=state.a.new_zeros((1,9));bw=state.a.new_zeros((1,5));ba=torch.zeros_like(bw)
    loss[0,0],bw[0,0],ba[0,0]=state.loss,state.bw,state.ba
    boundary=torch.stack((state.loss,state.bw,state.ba)).repeat(2,1)
    return CellTerm(angles,loss,bw,ba,boundary,state.panel.objective_weight)


def _square_linear(r):
    """[batch, 3, tokens] -> real degree-four coefficients (upper terms zero)."""
    a,b,c=r.unbind(1)
    v=torch.stack(((a*a+(b*b+c*c)/2).sum(-1), (2*a*b).sum(-1),
                   (2*a*c).sum(-1), ((b*b-c*c)/2).sum(-1), (b*c).sum(-1)), -1)
    return torch.cat((v,torch.zeros_like(v[...,:4])), -1)


def _codes(coeff, theta, scale_coeff, spec):
    # Paired member/angle arrays; broadcasting also handles one whole group.
    values=coeff[...,0]+coeff[...,1]*theta.cos()+coeff[...,2]*theta.sin()
    scale=(scale_coeff[0]+scale_coeff[1]*theta.cos()+scale_coeff[2]*theta.sin()).clamp_min(0)
    return (values/torch.where(scale>0,scale,torch.ones_like(scale))).round().clamp(spec.qmin,spec.qmax)


def _values(coeff,theta):
    return coeff[...,0]+coeff[...,1]*theta.cos()+coeff[...,2]*theta.sin()


def _actual_group(group,theta,xh):
    # Match direct rotate -> absmax -> divide -> round order at boundaries.
    values=_values(group.coeff,theta)
    dynamic=values.abs().max()/group.spec.qmax
    scale=dynamic if group.fixed_scale is None else group.fixed_scale
    code=(values/torch.where(scale>0,scale,torch.ones_like(scale))).round().clamp(group.spec.qmin,group.spec.qmax)
    return (code*scale)@xh,dynamic


def _group_projection(group, raw_angles, members, row_angles, xh, delta, boundary_delta,
                      risk_delta, risk_boundary_delta, norm, work, batch_size):
    """Accumulate one group's polynomial jumps and exact-rounding corrections.

    Inside an owner interval, W_hat(theta)=s(theta)*C. C@X changes by
    sum(delta_C[j]*X[j,:]) at code events. Rebase at chunk starts to bound
    prefix-sum drift; these are per-chunk group GEMVs, never per-event GEMMs.
    """
    cuts,inverse=torch.unique(raw_angles,sorted=True,return_inverse=True)
    mid=cuts[:-1]+(cuts[1:]-cuts[:-1])/2
    mapping=torch.searchsorted(row_angles,cuts)
    owner_cuts=torch.unique(inverse[members<0],sorted=True)
    owner_indices=owner_cuts.detach().cpu().tolist()  # scheduling only
    valid=members>=0
    # Simultaneous/duplicate incidences must update each (boundary, member) once.
    packed=torch.unique(inverse[valid]*len(group.coeff)+members[valid],sorted=True)
    event,member=packed//len(group.coeff),packed%len(group.coeff)
    previous_poly=None;previous_risk=None;rebases=0
    for begin,end in zip(owner_indices[:-1],owner_indices[1:]):
        work.check()
        sc=group.scale_coeff(mid[begin]);dynamic=group.scale_coeff(mid[begin],dynamic=True)
        owner=(_values(group.coeff,mid[begin])).abs().argmax()
        owner_coeff=group.coeff[owner]
        rc=scale_square(dynamic,len(group.coeff))/norm
        internal=(event>begin)&(event<end)
        ev,mi=event[internal],member[internal]
        before=_codes(group.coeff[mi],mid[ev-1],sc,group.spec)
        after=_codes(group.coeff[mi],mid[ev],sc,group.spec)
        change=after-before
        # Boundary values use the actual nearest-even operation at the event.
        actual_dynamic=_values(owner_coeff,cuts[ev]).abs()/group.spec.qmax
        actual_scale=actual_dynamic if group.fixed_scale is None else group.fixed_scale.expand_as(actual_dynamic)
        values=_values(group.coeff[mi],cuts[ev])
        boundary_code=(values/torch.where(actual_scale>0,actual_scale,torch.ones_like(actual_scale))).round().clamp(group.spec.qmin,group.spec.qmax)
        boundary_change=boundary_code-after
        for lo in range(begin,end,batch_size):
            work.check();hi=min(lo+batch_size,end);size=hi-lo
            current_code=_codes(group.coeff,mid[lo],sc,group.spec)
            initial=current_code@xh;rebases+=1
            selected=(ev>lo)&(ev<hi)
            jumps=xh.new_zeros((size,xh.shape[1]))
            jumps.index_add_(0,ev[selected]-lo,change[selected,None]*xh[mi[selected]])
            projected=initial[None,:]+jumps.cumsum(0)
            poly=projected[:,None,:]*sc[None,:,None]
            first=poly[:1] if previous_poly is None else poly[:1]-previous_poly[None,:,:]
            delta.index_add_(0,mapping[lo:hi],torch.cat((first,poly[1:]-poly[:-1]),0))
            previous_poly=poly[-1].clone()
            # Only owner changes alter the risk polynomial, even for fixed-scale decoding.
            if lo==begin:
                risk_delta[mapping[lo]]+=rc if previous_risk is None else rc-previous_risk
                previous_risk=rc.clone()
            all_theta=cuts[lo:hi]
            physical_dynamic=_values(owner_coeff,all_theta).abs()/group.spec.qmax
            physical_scale=physical_dynamic if group.fixed_scale is None else group.fixed_scale.expand_as(physical_dynamic)
            polynomial_scale=sc[0]+sc[1]*all_theta.cos()+sc[2]*all_theta.sin()
            corrections=(physical_scale-polynomial_scale)[:,None]*projected
            selected=(ev>lo)&(ev<hi)
            theta=cuts[ev[selected]]
            scale=actual_scale[selected]
            corrections.index_add_(0,ev[selected]-lo,
                                   (boundary_change[selected]*scale)[:,None]*xh[mi[selected]])
            # At owner/chunk boundaries recompute the whole group, including ties.
            at=cuts[lo];actual,actual_risk=_actual_group(group,at,xh)
            corrections[0]=actual-(poly[0]*basis(at,poly)[:,None]).sum(0)
            boundary_delta.index_add_(0,mapping[lo:hi],corrections)
            risk_correction=physical_dynamic.square()*len(group.coeff)/4/norm-trig_value(rc,all_theta)
            risk_correction[0]=actual_risk.square()*len(group.coeff)/4/norm-trig_value(rc,at)
            risk_boundary_delta.index_add_(0,mapping[lo:hi],risk_correction)
    # Last endpoint uses the final cell as its polynomial reference.
    at=cuts[-1];actual,actual_risk=_actual_group(group,at,xh)
    boundary_delta[mapping[-1]]+=actual-(previous_poly*basis(at,previous_poly)[:,None]).sum(0)
    risk_boundary_delta[mapping[-1]]+=actual_risk.square()*len(group.coeff)/4/norm-trig_value(previous_risk,at)
    return {'code_event_updates':len(packed),'group_projection_rebases':rebases}


def output_terms(orbit, events, work, batch_size, emit):
    """Independent output-row terms, preserving the complete hidden/FFN axes."""
    state=orbit.state;like=state.a;terms=[]
    for row in range(2):
        work.check()
        gids=[i for i,g in enumerate(orbit.groups) if g.row==row]
        angles=torch.unique(torch.cat([events[i][0] for i in gids]),sorted=True)
        count=len(angles)-1;tokens=state.x.shape[1]
        emit('batched_output_coefficients',force=True,panel_id=state.panel.panel_id,
             output_row=row,local_cell_count=count)
        delta=like.new_zeros((count,3,tokens));boundary_delta=like.new_zeros((count+1,tokens))
        rd=like.new_zeros((count,5));rb=like.new_zeros(count+1)
        for gid in gids:
            g=orbit.groups[gid]
            stats=_group_projection(g,*events[gid],angles,
                state.xh[g.start:g.start+len(g.coeff)],delta,boundary_delta,rd,rb,state.wnorm,work,batch_size)
            for key,value in stats.items():orbit.update_stats[key]=orbit.update_stats.get(key,0)+value
        # Sum contributions of all groups in this row before taking its square.
        poly=delta.cumsum(0);del delta
        risks=rd.cumsum(0);del rd
        lc=like.new_empty((count,9));boundary=like.new_empty((count+1,3));boundary[:,2]=0
        target=state.target[orbit.pair[row]];partner=state.target[orbit.pair[1-row]]
        sign=1 if row==0 else -1
        for lo in range(0,count+1,batch_size):
            work.check();hi=min(lo+batch_size,count+1)
            ix=torch.arange(lo,hi,device=like.device).clamp_max(count-1)
            theta=angles[lo:hi]
            prediction=(poly[ix]*torch.stack((torch.ones_like(theta),theta.cos(),theta.sin()),1)[:,:,None]).sum(1)
            error=prediction+boundary_delta[lo:hi]-theta.cos()[:,None]*target+sign*theta.sin()[:,None]*partner
            boundary[lo:hi,0]=error.square().sum(1)/state.denominator
            boundary[lo:hi,1]=trig_value(risks[ix],theta)+rb[lo:hi]
            if lo<count:
                stop=min(hi,count);residual=poly[lo:stop].clone()
                residual[:,1]-=target;residual[:,2]+=sign*partner
                lc[lo:stop]=_square_linear(residual)/state.denominator
        terms.append(CellTerm(angles,lc,risks,torch.zeros_like(risks),boundary,state.panel.objective_weight))
        del poly,boundary_delta,rb
    # Unchanged rows and activations are constant terms, included exactly once.
    remainder=state.loss-state.error[list(orbit.pair)].square().sum()/state.denominator
    risk_rows=state.a[list(orbit.pair)].reshape(2,-1,state.wg).abs().amax(-1)/state.wspec.qmax
    remaining_risk=state.bw-risk_rows.square().sum()*state.wg/4/state.wnorm
    term=terms[0];term.loss[:,0]+=remainder;term.bw[:,0]+=remaining_risk;term.ba[:,0]+=state.ba
    term.boundary+=torch.stack((remainder,remaining_risk,state.ba))
    return terms


def constant_scale_input_term(orbit,angles,work,emit,batch_size):
    """Exact rank-two updates for fixed decoding scales or radius-stable AbsMax.

    Unrotated codes then stay fixed. The full hidden-axis contribution is
    retained in base_error; each angle only changes two outer products.
    Dynamic decoding requires that bound for every affected group. Fixed-scale
    decoding instead gets its changing *physical* risks from input_risks().
    """
    state=orbit.state;pair=list(orbit.pair);like=state.a;count=len(angles)-1
    w=state.a[:,pair];x=state.x[pair]
    sw=state.sw if state.fixed_w is None else state.fixed_w
    sx=state.sx if state.fixed_a is None else state.fixed_a
    wi=torch.tensor(pair,device=like.device)//state.wg;ai=torch.tensor(pair,device=like.device)//state.ag
    ws=sw[:,wi,0];xs=sx[:,ai,0].T
    base_error=state.error-state.ah[:,pair]@state.xh[pair]
    def losses(theta):
        c,s=theta.cos()[:,None],theta.sin()[:,None]
        wr=torch.stack((c*w[:,0]-s*w[:,1],s*w[:,0]+c*w[:,1]),-1)
        xr=torch.stack((c*x[0]-s*x[1],s*x[0]+c*x[1]),1)
        wh=(wr/torch.where(ws>0,ws,torch.ones_like(ws))).round().clamp(state.wspec.qmin,state.wspec.qmax)*ws
        xh=(xr/torch.where(xs>0,xs,torch.ones_like(xs))).round().clamp(state.aspec.qmin,state.aspec.qmax)*xs
        return (base_error+wh@xh).square().sum((1,2))/state.denominator
    loss=like.new_zeros((count,9));bw=like.new_zeros((count,5));ba=torch.zeros_like(bw)
    bw[:,0]=state.bw;ba[:,0]=state.ba
    boundary=like.new_empty((count+1,3));boundary[:,1]=state.bw;boundary[:,2]=state.ba
    for lo in range(0,count+1,batch_size):
        work.check();hi=min(lo+batch_size,count+1)
        emit('batched_constant_scale_input',panel_id=state.panel.panel_id,local_cell_index=lo,local_cell_count=count,output_row=None)
        boundary[lo:hi,0]=losses(angles[lo:hi])
        stop=min(hi,count)
        if lo<stop:loss[lo:stop,0]=losses(angles[lo:stop]+(angles[lo+1:stop+1]-angles[lo:stop])/2)
    orbit.update_stats['constant_scale_input_rank_two_cells']=count
    return CellTerm(angles,loss,bw,ba,boundary,state.panel.objective_weight)


def input_risks(orbit,events,angles,work):
    """Physical dynamic-scale risks; frozen decoder scales never substitute."""
    like=angles;count=len(angles)-1
    coefficients=[like.new_zeros((count,5)),like.new_zeros((count,5))]
    corrections=like.new_zeros((count+1,2))
    for gid,g in enumerate(orbit.groups):
        if g.stable:continue
        work.check();side=0 if g.side=='w' else 1
        norm=orbit.state.wnorm if side==0 else orbit.state.anorm
        cuts=torch.unique(events[gid][0][events[gid][1]<0],sorted=True)
        indices=torch.searchsorted(angles,cuts)
        initial=_values(g.coeff,like.new_zeros(())).abs().max()/g.spec.qmax
        previous=like.new_zeros(5);previous[0]=initial.square()*len(g.coeff)/4/norm
        for i in range(len(cuts)-1):
            lo,hi=cuts[i],cuts[i+1];sc=g.scale_coeff(lo+(hi-lo)/2,dynamic=True)
            rc=scale_square(sc,len(g.coeff))/norm
            coefficients[side][indices[i]]+=rc-previous;previous=rc
            physical=_values(g.coeff,lo).abs().max()/g.spec.qmax
            corrections[indices[i],side]+=physical.square()*len(g.coeff)/4/norm-trig_value(rc,lo)
        physical=_values(g.coeff,cuts[-1]).abs().max()/g.spec.qmax
        corrections[indices[-1],side]+=physical.square()*len(g.coeff)/4/norm-trig_value(previous,cuts[-1])
    for c,base in zip(coefficients,(orbit.state.bw,orbit.state.ba)):
        c.copy_(c.cumsum(0));c[:,0]+=base
    ix=torch.arange(count+1,device=like.device).clamp_max(count-1)
    boundary=torch.stack([trig_value(c[ix],angles) for c in coefficients],-1)+corrections
    return *coefficients,boundary


def input_boundary(orbit,theta,dirty):
    """Pointwise row/column updates for full-row groups, without changing cells.

    W_new = (s_new/s_old) W_old + s_new (C_new-C_old).
    The first product is already cached in the current residual. Only changed
    codes need outer-product updates. Process weights before activations and
    use updated weight rows in activation updates, retaining the cross term.
    """
    state=orbit.state;b=basis(theta,state.a)
    e=orbit.error[0]+orbit.error[1]*theta.cos()+orbit.error[2]*theta.sin()+orbit.error[3]*(2*theta).cos()+orbit.error[4]*(2*theta).sin()
    prediction=e+state.target
    risks=[trig_value(orbit.bw,theta),trig_value(orbit.ba,theta)]
    changed_weights={}
    # Group metadata are host scheduling; numerical values never leave CUDA.
    for gid,_ in sorted(dirty,key=lambda item:orbit.groups[item[0]].side!='w'):
        g=orbit.groups[gid];weight=g.side=='w';sc=orbit.group_scales[gid]
        physical=_values(g.coeff,theta);dynamic=physical.abs().max()/g.spec.qmax
        scale=dynamic if g.fixed_scale is None else g.fixed_scale
        actual_code=(physical/torch.where(scale>0,scale,torch.ones_like(scale))).round().clamp(g.spec.qmin,g.spec.qmax)
        coefficients=orbit.aw[:,g.row,:] if weight else orbit.ax[:,:,g.row]
        component=sc.abs().argmax();den=sc[component]
        old_code=(coefficients[component]/torch.where(den!=0,den,torch.ones_like(den))).round()
        dc=actual_code-old_code;changed=torch.where(dc!=0)[0]
        reference=(sc@b).clamp_min(0)
        if bool(reference>0):
            opposite=(orbit.ax[:,changed,:]*b[:,None,None]).sum(0) if weight else (orbit.aw[:,:,changed]*b[:,None,None]).sum(0)
            if not weight:
                for row,values in changed_weights.items():opposite[row]=values[changed]
            if weight:prediction[g.row]=prediction[g.row]*(scale/reference)+scale*(dc[changed]@opposite)
            else:prediction[:,g.row]=prediction[:,g.row]*(scale/reference)+scale*(opposite@dc[changed])
        else:
            # Degenerate zero-scale boundary: exact fallback, explicitly counted.
            opposite=(orbit.ax*b[:,None,None]).sum(0) if weight else (orbit.aw*b[:,None,None]).sum(0)
            if not weight:
                for row,values in changed_weights.items():opposite[row]=values
            if weight:prediction[g.row]=(actual_code*scale)@opposite
            else:prediction[:,g.row]=opposite@(actual_code*scale)
            orbit.update_stats['boundary_full_group_fallbacks']=orbit.update_stats.get('boundary_full_group_fallbacks',0)+1
        if weight:changed_weights[g.row]=actual_code*scale
        old_risk=getattr(g,'_risk_scale',sc)
        side=0 if weight else 1;norm=state.wnorm if weight else state.anorm
        risks[side]+=(dynamic.square()*len(g.coeff)/4-trig_value(scale_square(old_risk,len(g.coeff)),theta))/norm
    orbit.update_stats['pointwise_boundary_updates']=orbit.update_stats.get('pointwise_boundary_updates',0)+len(dirty)
    return torch.stack(((prediction-state.target).square().sum()/state.denominator,*risks))


def input_term(orbit, events, work, emit, batch_size=1024):
    """Bilateral incremental updates only at this panel's events (no dense GEMM)."""
    from .tensor_solver import EventIndex
    like=orbit.state.a
    raw=torch.cat([e[0] for e in events]);members=torch.cat([e[1] for e in events])
    keys=torch.cat([torch.full((len(e[0]),),i,device=like.device,dtype=torch.long) for i,e in enumerate(events)])
    angles,inverse=torch.unique(raw,sorted=True,return_inverse=True)
    stable=all(g.stable for g in orbit.groups)
    if stable or (orbit.state.fixed_w is not None and orbit.state.fixed_a is not None):
        term=constant_scale_input_term(orbit,angles,work,emit,batch_size)
        if not stable:
            term.bw,term.ba,term.boundary[:,1:]=input_risks(orbit,events,angles,work)
        return term
    from .tensor_input_cells import can_batch, full_row_input_term
    if can_batch(orbit):
        return full_row_input_term(orbit,angles,work,emit,batch_size)
    index=EventIndex(inverse,keys,members,len(angles))
    count=len(angles)-1
    loss=like.new_empty((count,9));bw=like.new_empty((count,5));ba=like.new_empty((count,5))
    boundary=like.new_empty((count+1,3))
    def incident(n):
        if n==0 or n==count:return [(gid,None) for gid in range(len(orbit.groups))]
        k,m=index.at(n);result=[]
        for gid in torch.unique(k):
            ids=m[k==gid];result.append((int(gid),None if bool((ids<0).any()) else torch.unique(ids)))
        return result
    for n in range(count):
        work.check();emit('input_coefficients',panel_id=orbit.state.panel.panel_id,local_cell_index=n,local_cell_count=count,output_row=None)
        mid=angles[n]+(angles[n+1]-angles[n])/2
        for gid,ids in incident(n):orbit.replace(gid,mid,indices=ids)
        loss[n],bw[n],ba[n]=orbit.loss_coeff,orbit.bw,orbit.ba
        for b in ([0,1] if n==0 else [n+1]):
            dirty=incident(b)
            if orbit.state.wg==orbit.state.a.shape[1] and orbit.state.ag==orbit.state.x.shape[0]:
                boundary[b]=input_boundary(orbit,angles[b],dirty)
            else:
                for gid,_ in dirty:orbit.replace(gid,angles[b],actual=True)
                boundary[b]=torch.stack([trig_value(c,angles[b]) for c in (orbit.loss_coeff,orbit.bw,orbit.ba)])
                for gid,_ in dirty:orbit.replace(gid,mid)
    return CellTerm(angles,loss,bw,ba,boundary,orbit.state.panel.objective_weight)


def _root_possible(coeff,left,right):
    """Keep cells where a root cannot be excluded by midpoint Taylor bounds."""
    half_width = (right - left) / 2
    midpoint = left + half_width
    cosine, sine = coeff[..., 1::2], coeff[..., 2::2]
    k = torch.arange(1, cosine.shape[-1] + 1, device=coeff.device, dtype=coeff.dtype)
    phase = midpoint[..., None] * k
    cos_mid, sin_mid = phase.cos(), phase.sin()
    value = coeff[..., 0] + (cosine * cos_mid + sine * sin_mid).sum(-1)
    slope = (k * (sine * cos_mid - cosine * sin_mid)).sum(-1)
    amplitude = cosine.abs() + sine.abs()
    max_slope = (k * amplitude).sum(-1)
    max_curvature = (k.square() * amplitude).sum(-1)
    # Across this cell, |f-f(mid)| <= h |f'(mid)| + h^2 sup|f''|/2.
    # Small f'(mid) is common when harmonic terms cancel each other.
    bound = half_width * torch.minimum(max_slope, slope.abs() + half_width * max_curvature / 2)
    norm = coeff.abs().sum(-1)
    scale = norm + half_width * max_slope + half_width.square() * max_curvature
    guard = 64 * torch.finfo(coeff.dtype).eps * scale.clamp_min(torch.finfo(coeff.dtype).tiny)
    return (norm > 0) & (value.abs() <= bound + guard)


def search_terms(terms,angles,baseline,limits,config,work,trace,emit,batch_size):
    """Merge every panel before root/candidate selection; one angle for all sites."""
    like=angles;short_theta=like.new_empty(0);short_loss=like.new_empty(0)
    candidates=0;root_cells=0
    def feasible(bw,ba):
        if not config.constrain_budgets:return torch.ones_like(bw,dtype=torch.bool)
        return (bw<=limits['budget_w']+config.budget_tolerance*limits['budget_w'].clamp_min(1)) & (ba<=limits['budget_a']+config.budget_tolerance*limits['budget_a'].clamp_min(1))
    def consider(theta,loss,bw,ba):
        nonlocal short_theta,short_loss,candidates
        theta,loss,bw,ba=[x.reshape(-1) for x in (theta,loss,bw,ba)];candidates+=theta.numel()
        score=loss.masked_fill(~(feasible(bw,ba)&(loss<baseline['loss'])),torch.inf)
        values=torch.cat((short_loss,score));points=torch.cat((short_theta,theta))
        short_loss,keep=values.topk(min(32,len(values)),largest=False);short_theta=points[keep]
    for begin in range(0,len(angles)-1,batch_size):
        work.check();end=min(begin+batch_size,len(angles)-1)
        emit('batched_candidates',cell_index=begin,panel_id=None,output_row=None,local_cell_index=None,local_cell_count=None)
        lo,hi=angles[begin:end],angles[begin+1:end+1];mid=lo+(hi-lo)/2
        lc=like.new_zeros((end-begin,9));bw=like.new_zeros((end-begin,5));ba=torch.zeros_like(bw)
        points=angles[begin:end+1];boundary=like.new_zeros((len(points),3))
        for term in terms:
            ix=(torch.searchsorted(term.angles,mid,right=True)-1).clamp(0,len(term.loss)-1)
            lc+=term.weight*term.loss[ix];bw+=term.weight*term.bw[ix];ba+=term.weight*term.ba[ix]
            bi=torch.searchsorted(term.angles,points,right=True)-1;ci=bi.clamp(0,len(term.loss)-1)
            measured=torch.stack([trig_value(c[ci],points) for c in (term.loss,term.bw,term.ba)],-1)
            # Only incident boundaries need stored actual-rounding values.
            exact=term.angles[bi]==points
            boundary+=term.weight*torch.where(exact[:,None],term.boundary[bi],measured)
        derivative=torch.zeros_like(lc);k=torch.arange(1,5,device=like.device,dtype=like.dtype)
        derivative[:,1::2]=k*lc[:,2::2];derivative[:,2::2]=-k*lc[:,1::2]
        possible=_root_possible(derivative,lo,hi)
        if config.constrain_budgets:
            for c,key in [(bw,'budget_w'),(ba,'budget_a')]:
                shifted=c.clone();shifted[:,0]-=limits[key];possible|=_root_possible(shifted,lo,hi)
        # Shared candidate points for root-free cells, computed in one batch.
        simple=~possible
        sample=torch.stack((mid,torch.nextafter(lo,hi),torch.nextafter(hi,lo),lo+(hi-lo)*1e-7,hi-(hi-lo)*1e-7),-1)[simple]
        consider(sample,*[trig_value(c[simple,None,:],sample) for c in (lc,bw,ba)])
        # Near-zero, tangent and root-containing cells retain the numerical QR path.
        for i in torch.where(possible)[0].detach().cpu().tolist():
            work.check();root_cells+=1
            stationary=trig_roots(derivative[i],lo[i],hi[i]);cuts=[torch.stack((lo[i],hi[i]))]
            if config.constrain_budgets:
                for c,key in [(bw[i], 'budget_w'),(ba[i], 'budget_a')]:
                    shifted=c.clone();shifted[0]-=limits[key];cuts.append(trig_roots(shifted,lo[i],hi[i]))
            cuts=torch.unique(torch.cat(cuts),sorted=True);a,b=cuts[:-1],cuts[1:]
            sample=torch.cat((stationary,cuts[1:-1],a+(b-a)/2,torch.nextafter(a,b),torch.nextafter(b,a),a+(b-a)*1e-7,b-(b-a)*1e-7))
            consider(sample,*[trig_value(c[i],sample) for c in (lc,bw,ba)])
        # Adjacent chunks share one endpoint; evaluate it once in the count.
        first=0 if begin==0 else 1
        consider(points[first:],*boundary[first:].unbind(-1))
        trace['cells_completed']=end
    finite=torch.isfinite(short_loss)
    return short_theta[finite],short_loss[finite],{'candidate_count':candidates,'boundary_rounding_evaluations':len(angles),
        'root_candidate_cells':root_cells,'batch_size':batch_size}
