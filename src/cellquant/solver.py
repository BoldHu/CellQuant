"""Shared global-R1 solver with complete groups and incremental event sweeps.

Dense panel transforms may use torch CUDA. Event envelopes and degree-eight
numerical roots run in FP64 on CPU. Each code/owner event updates only a product
row/column and its loss coefficients, never a full panel GEMM. Whole-row output
orbits retain all hidden rows in the state and restrict *updates* to the pair.
Wall-clock/event limits stop incomplete searches explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import math
import os
import tempfile
import time
import hashlib
import numpy as np

from .events import GroupOrbit, ResourceLimit, WorkBudget
from .math_reference import _trig_roots, _real_to_fourier, _fourier_to_real, trig_value
from .quantization import QuantSpec, quantize, risk
from .runtime import atomic_json, config_hash
from .progress import report


@dataclass
class CalibrationPanel:
    panel_id: str
    placement: str
    weight: np.ndarray
    activation: np.ndarray  # column convention: reduction x tokens
    token_ids_hash: str = ""
    complete_group_width: int | None = None
    objective_weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        tensor = hasattr(self.weight, 'detach')
        if not tensor:
            self.weight = np.asarray(self.weight)
            self.activation = np.asarray(self.activation)
        if self.placement not in {"input", "output"}:
            raise ValueError("placement must be input (WQ.T/QX) or output (QB/U)")
        if (self.weight.ndim != 2 or self.activation.ndim != 2 or
                self.weight.shape[1] != self.activation.shape[0] or
                min(*self.weight.shape, *self.activation.shape) <= 0):
            raise ValueError("Panel requires complete W[m,d]/X[d,T] or B[d,f]/U[f,T]")
        if tensor:
            import torch
            finite = bool(torch.isfinite(self.weight).all() & torch.isfinite(self.activation).all())
            if self.weight.device != self.activation.device:raise ValueError('Panel tensors must share a device')
        else:finite = np.isfinite(self.weight).all() and np.isfinite(self.activation).all()
        if not finite:
            raise ValueError("Nonfinite panel")
        if not math.isfinite(self.objective_weight) or self.objective_weight <= 0:
            raise ValueError("Positive finite objective weight required")
        if self.complete_group_width is None:
            self.complete_group_width = self.weight.shape[1]
        if self.complete_group_width != self.weight.shape[1]:
            raise ValueError("Truncated reduction channels are forbidden")

    @property
    def hidden_size(self):
        return self.weight.shape[1 if self.placement == "input" else 0]


@dataclass
class SolverConfig:
    backend: str = "numpy_reference"
    compute_dtype: str = "float64"
    theta_limit: float = math.pi/4
    event_limit_per_pair: int = 200000
    seconds_per_pair: float = 300.
    wall_seconds: float = 3600.
    screen: bool = True
    constrain_budgets: bool = True
    fixed_scales: bool = False
    budget_tolerance: float = 1e-10
    device: str = "cpu"
    checkpoint_path: str | None = None
    resume: bool = False
    execution_fingerprint: str | None = None


def _gemm(a, b, device="cpu"):
    if device == "cpu":
        return a @ b
    import torch
    with torch.inference_mode():
        result = torch.as_tensor(a, device=device) @ torch.as_tensor(b, device=device)
        torch.cuda.synchronize()
        return result.cpu().numpy()


def hadamard_rotation(d, seed=0):
    from .panels import normalized_hadamard_last_axis
    rng = np.random.default_rng(seed)
    return normalized_hadamard_last_axis(np.eye(d))*rng.choice([-1., 1.], size=d)[None, :]


def pair_schedule(d, count, seed):
    if d < 2 or count < 0:
        raise ValueError("Invalid dimension/pair count")
    rng = np.random.default_rng(seed)
    return [tuple(map(int, rng.choice(d, 2, replace=False))) for _ in range(count)]


def rotate_rows(z, pair, theta):
    i, j = pair
    c, s = math.cos(theta), math.sin(theta)
    u, v = z[i].copy(), z[j].copy()
    z[i], z[j] = c*u-s*v, s*u+c*v


def product_coeff(a, x):
    """Coefficient-first arrays: three linear -> five product coefficients."""
    return np.stack([a[0]@x[0]+(a[1]@x[1]+a[2]@x[2])/2,
                     a[0]@x[1]+a[1]@x[0], a[0]@x[2]+a[2]@x[0],
                     (a[1]@x[1]-a[2]@x[2])/2,
                     (a[1]@x[2]+a[2]@x[1])/2])


def square_coeff(residual):
    fourier = _real_to_fourier(np.moveaxis(residual, 0, -1)).reshape(-1, 5)
    conv = np.zeros(9, dtype=np.complex128)
    for i in range(5):
        for j in range(5):
            conv[i+j] += np.sum(fourier[:, i]*fourier[:, j])
    return _fourier_to_real(conv)


def scale_square(sc, width):
    a,b,c = sc
    return np.array([a*a+(b*b+c*c)/2,2*a*b,2*a*c,(b*b-c*c)/2,b*c])*width/4


class PanelState:
    def __init__(self, panel, q, wspec, aspec, *, device="cpu", fixed=False):
        self.panel, self.wspec, self.aspec = panel, wspec, aspec
        self.wg, self.ag = wspec.width(panel.weight.shape[1]), aspec.width(panel.activation.shape[0])
        w, x = panel.weight.astype(np.float64), panel.activation.astype(np.float64)
        identity = np.array_equal(q,np.eye(len(q)))
        self.a = w.copy() if identity else (_gemm(w, q.T, device) if panel.placement == "input" else _gemm(q, w, device))
        self.x = x.copy() if identity or panel.placement == "output" else _gemm(q,x,device)
        self.target = _gemm(w, x, device)
        if panel.placement == "output" and not identity:
            self.target = _gemm(q, self.target, device)
        self.denominator = max(float(np.square(self.target).sum()), 1e-30)
        self.wnorm = max(float(np.square(w).sum()), 1e-30)
        self.anorm = max(float(np.square(x).sum()), 1e-30)
        self.fixed_w = quantize(self.a, wspec)[2] if fixed else None
        self.fixed_a = quantize(self.x.T, aspec)[2] if fixed else None
        self.refresh()

    def refresh(self):
        self.ah, _, self.sw = quantize(self.a, self.wspec, scales=self.fixed_w)
        xh, _, self.sx = quantize(self.x.T, self.aspec, scales=self.fixed_a)
        self.xh = xh.T
        self.error = self.ah @ self.xh-self.target
        self.loss = float(np.square(self.error).sum()/self.denominator)
        # Physical dynamic risk is always checked, also for fixed-scale control.
        self.bw = risk(quantize(self.a, self.wspec)[2], self.wg)/self.wnorm
        self.ba = risk(quantize(self.x.T, self.aspec)[2], self.ag)/self.anorm

    def accept(self, pair, theta):
        if self.panel.placement == "input":
            rotate_rows(self.a.T, pair, theta)
            rotate_rows(self.x, pair, theta)
        else:
            rotate_rows(self.a, pair, theta)
            rotate_rows(self.target, pair, theta)
        self.refresh()

    def direct(self, pair, theta):
        if self.panel.placement == "output":
            # Unchanged output rows contribute exactly their cached error and
            # risk; all reduction channels in the two changed rows are kept.
            rows=list(pair);a=self.a[rows].copy();y=self.target[rows].copy()
            rotate_rows(a,(0,1),theta);rotate_rows(y,(0,1),theta)
            frozen=None if self.fixed_w is None else self.fixed_w[rows]
            ah=quantize(a,self.wspec,scales=frozen)[0]
            new_error=ah@self.xh-y
            loss=self.loss+(float(np.square(new_error).sum())-float(np.square(self.error[rows]).sum()))/self.denominator
            old_risk=risk(quantize(self.a[rows],self.wspec)[2],self.wg)
            new_risk=risk(quantize(a,self.wspec)[2],self.wg)
            return {"loss":loss,"budget_w":self.bw+(new_risk-old_risk)/self.wnorm,"budget_a":self.ba}
        a, x, y = self.a.copy(), self.x.copy(), self.target.copy()
        if self.panel.placement == "input":
            rotate_rows(a.T, pair, theta)
            rotate_rows(x, pair, theta)
        else:
            rotate_rows(a, pair, theta)
            rotate_rows(y, pair, theta)
        ah = quantize(a, self.wspec, scales=self.fixed_w)[0]
        xh = quantize(x.T, self.aspec, scales=self.fixed_a)[0].T
        return {"loss":float(np.square(ah@xh-y).sum()/self.denominator),
                "budget_w":risk(quantize(a,self.wspec)[2],self.wg)/self.wnorm,
                "budget_a":risk(quantize(x.T,self.aspec)[2],self.ag)/self.anorm}


class PanelOrbit:
    """Incremental analytic state for one pair; output updates are only two rows."""
    def __init__(self, state, pair):
        self.state, self.pair = state, pair
        output = state.panel.placement == "output"
        i,j = pair
        a = state.a[[i,j]] if output else state.a
        ah = state.ah[[i,j]] if output else state.ah
        self.aw = np.zeros((3,*a.shape))
        self.ax = np.zeros((3,*state.x.shape))
        self.aw[0], self.ax[0] = ah, state.xh
        self.error = np.zeros((5,a.shape[0],state.x.shape[1]))
        self.error[0] = state.error[[i,j]] if output else state.error
        self.loss_coeff = square_coeff(self.error)/state.denominator
        if output:
            self.loss_coeff[0] += state.loss-float(np.square(self.error[0]).sum()/state.denominator)
            # Subtract G(theta)*target, restore target(theta=0) in constant term.
            target = state.target[[i,j]]
            self.error[0] += target
            self.error[1] -= target
            self.error[2,0] += target[1]
            self.error[2,1] -= target[0]
            self.loss_coeff = square_coeff(self.error)/state.denominator
            self.loss_coeff[0] += state.loss-float(np.square(state.error[[i,j]]).sum()/state.denominator)
        self.bw = np.array([state.bw,0.,0.,0.,0.])
        self.ba = np.array([state.ba,0.,0.,0.,0.])
        self.groups = []
        self.group_scales = []
        self.update_stats = {"code_member_updates":0, "owner_member_updates":0,
                             "row_column_updates":0, "full_gemm_per_event":0}
        if output:
            for row in range(2):
                coeff = np.zeros((a.shape[1],3))
                coeff[:,1] = a[row]
                coeff[:,2] = -a[1] if row == 0 else a[0]
                for start in range(0,a.shape[1],state.wg):
                    g = start//state.wg
                    fixed = None if state.fixed_w is None else float(state.fixed_w[pair[row],g,0])
                    self._add(GroupOrbit("w",row,start,coeff[start:start+state.wg].copy(),state.wspec,False,fixed),
                              state.sw[pair[row],g,0])
        else:
            for side, vectors, spec, g, scales, frozen in (
                    ("w",state.a,state.wspec,state.wg,state.sw,state.fixed_w),
                    ("a",state.x.T,state.aspec,state.ag,state.sx,state.fixed_a)):
                for row, vec in enumerate(vectors):
                    radius = math.hypot(vec[i],vec[j])
                    for start in sorted({i//g*g,j//g*g}):
                        coeff = np.zeros((g,3)); coeff[:,0] = vec[start:start+g]
                        for index, b,c in [(i,vec[i],-vec[j]),(j,vec[j],vec[i])]:
                            if start <= index < start+g:
                                coeff[index-start] = [0,b,c]
                        changed = np.any(coeff[:,1:] != 0,axis=1)
                        if not changed.any():
                            continue
                        other = np.abs(coeff[~changed,0])
                        stable = bool(other.size and other.max() >= radius)
                        fixed = None if frozen is None else float(frozen[row,start//g,0])
                        self._add(GroupOrbit(side,row,start,coeff,spec,stable,fixed),scales[row,start//g,0])

    def _add(self, group, scale):
        self.groups.append(group)
        self.group_scales.append(np.array([float(scale),0.,0.]))

    def replace(self, gid, theta, *, actual=False, indices=None):
        group = self.groups[gid]
        old_scale = self.group_scales[gid]
        new_scale = group.scale_coeff(theta)
        owner_changed = not np.array_equal(new_scale,old_scale)
        if owner_changed or indices is None:
            ix = np.arange(len(group.coeff))
        else:
            ix = np.array(sorted(indices),dtype=int)
        if not len(ix):
            return None
        cols = group.start+ix
        dest = self.aw[:,group.row,cols].copy() if group.side == "w" else self.ax[:,cols,group.row].copy()
        replacement,_ = group.reconstruction(theta,actual=actual,indices=ix)
        replacement = replacement.T
        delta = replacement-dest
        active = np.any(delta != 0,axis=0)
        changed_cols = cols[active]
        if active.any():
            if group.side == "w":
                old_error = self.error[:,group.row:group.row+1].copy()
                diff = product_coeff(delta[:,None,active],self.ax[:,changed_cols,:])
                self.error[:,group.row:group.row+1] += diff
                self.aw[:,group.row,changed_cols] = replacement[:,active]
                new_error = self.error[:,group.row:group.row+1]
            else:
                old_error = self.error[:,:,group.row:group.row+1].copy()
                diff = product_coeff(self.aw[:,:,changed_cols],delta[:,active,None])
                self.error[:,:,group.row:group.row+1] += diff
                self.ax[:,changed_cols,group.row] = replacement[:,active]
                new_error = self.error[:,:,group.row:group.row+1]
            self.loss_coeff += (square_coeff(new_error)-square_coeff(old_error))/self.state.denominator
            self.update_stats["row_column_updates"] += 1
            key = "owner_member_updates" if owner_changed else "code_member_updates"
            self.update_stats[key] += int(active.sum())
        # Frozen-scale ablation uses dynamic risk constraints; obtain physical sc.
        if group.fixed_scale is None:
            risk_scale = new_scale
        else:
            dynamic = GroupOrbit(group.side,group.row,group.start,group.coeff,group.spec,group.stable)
            risk_scale = dynamic.scale_coeff(theta)
        old_risk = getattr(group,"_risk_scale",old_scale)
        budget = self.bw if group.side == "w" else self.ba
        norm = self.state.wnorm if group.side == "w" else self.state.anorm
        budget += (scale_square(risk_scale,len(group.coeff))-scale_square(old_risk,len(group.coeff)))/norm
        group._risk_scale = risk_scale.copy()
        self.group_scales[gid] = new_scale.copy()
        return (gid,cols,dest,old_scale.copy(),old_risk.copy())

    def snapshot(self):
        return (self.aw.copy(),self.ax.copy(),self.error.copy(),self.loss_coeff.copy(),self.bw.copy(),self.ba.copy())


def aggregate(states):
    return {key:sum(s.panel.objective_weight*getattr(s,attr) for s in states)
            for key,attr in [("loss","loss"),("budget_w","bw"),("budget_a","ba")]}


def solve_pair(states, pair, limits, config=SolverConfig()):
    if not 0 < config.theta_limit < math.pi:
        raise ValueError("theta_limit must lie in (0, pi)")
    d = states[0].panel.hidden_size
    if len(pair) != 2 or pair[0] == pair[1] or min(pair)<0 or max(pair)>=d:
        raise ValueError("Invalid Givens pair")
    start = time.perf_counter()
    work = WorkBudget(config.event_limit_per_pair,config.seconds_per_pair)
    orbit = [PanelOrbit(s,pair) for s in states]
    baseline = aggregate(states)
    left,right = -config.theta_limit,config.theta_limit
    events = {left:{},right:{},0.:{}}
    stats = {side:{"owner_intervals":0,"member_visits":0,"stable_groups":0,"groups":0} for side in ("input","output")}
    timings = {"event_generation":0.,"sort":0.,"coefficient_updates":0.,"roots":0.,"boundary_rounding":0.,"verification":0.}
    t = time.perf_counter()
    for p, orb in enumerate(orbit):
        for gid, group in enumerate(orb.groups):
            ev,st = group.enumerate(left,right,config.screen,work)
            for key,v in st.items(): stats[orb.state.panel.placement][key] += v
            for theta,ids in ev.items():
                bucket = events.setdefault(theta,{})
                bucket[(p,gid)] = ids
    timings["event_generation"] = time.perf_counter()-t
    t = time.perf_counter(); angles = sorted(events); timings["sort"] = time.perf_counter()-t
    def coefficients():
        return [sum(o.state.panel.objective_weight*getattr(o,k) for o in orbit) for k in ("loss_coeff","bw","ba")]
    def feasible(bw,ba):
        return (not config.constrain_budgets or (bw <= limits["budget_w"]+config.budget_tolerance*max(1.,limits["budget_w"])
                and ba <= limits["budget_a"]+config.budget_tolerance*max(1.,limits["budget_a"])))
    # Keep promising candidates, then re-evaluate complete quantizers at the end.
    import heapq
    shortlist=[]; candidate_count=0; boundary_count=0
    def consider(theta,loss,bw,ba):
        nonlocal candidate_count
        candidate_count += 1
        if feasible(bw,ba) and loss < baseline["loss"]:
            item=(-float(loss),float(theta))
            if len(shortlist)<32: heapq.heappush(shortlist,item)
            elif item>shortlist[0]: heapq.heapreplace(shortlist,item)
    for n,(lo,hi) in enumerate(zip(angles[:-1],angles[1:])):
        work.check()
        middle = lo+(hi-lo)/2
        t = time.perf_counter()
        dirty = events[lo]
        if n==0:
            dirty={(p,gid):None for p,o in enumerate(orbit) for gid in range(len(o.groups))}
        for (p,gid),ids in dirty.items():
            orbit[p].replace(gid,middle,indices=ids)
        timings["coefficient_updates"] += time.perf_counter()-t
        lc,bw,ba = coefficients()
        t = time.perf_counter()
        derivative=np.zeros(9)
        for k in range(1,5): derivative[2*k-1],derivative[2*k]=k*lc[2*k],-k*lc[2*k-1]
        stationary = _trig_roots(derivative,lo,hi)
        cuts=[lo,hi]
        if config.constrain_budgets:
            for c,key in [(bw,"budget_w"),(ba,"budget_a")]:
                c=c.copy();c[0]-=limits[key];cuts.extend(_trig_roots(c,lo,hi))
        cuts=sorted(set(cuts))
        candidates=list(stationary)+cuts[1:-1]
        for a,b in zip(cuts[:-1],cuts[1:]):
            candidates.extend([a+(b-a)/2,np.nextafter(a,b),np.nextafter(b,a),a+(b-a)*1e-7,b-(b-a)*1e-7])
        for theta in candidates:
            consider(theta,trig_value(lc,theta),trig_value(bw,theta),trig_value(ba,theta))
        timings["roots"] += time.perf_counter()-t
        # Evaluate true rounding at BOTH actual event boundaries. Save only the
        # incident group's values; undo via a new interior reconstruction.
        t=time.perf_counter()
        for theta in ([lo,hi] if n==0 else [hi]):
            incident=events[theta]
            if theta in (left,right):
                incident={(p,gid):None for p,o in enumerate(orbit) for gid in range(len(o.groups))}
            for (p,gid) in incident: orbit[p].replace(gid,theta,actual=True)
            l,b,c=coefficients()
            consider(theta,trig_value(l,theta),trig_value(b,theta),trig_value(c,theta))
            boundary_count+=1
            for (p,gid) in incident: orbit[p].replace(gid,middle)
        timings["boundary_rounding"]+=time.perf_counter()-t
    t=time.perf_counter();best=dict(baseline,theta=0.)
    # Real full-group quantization prevents polynomial drift/root error from
    # authorizing an invalid update. This is per finalist, not per event.
    for _,theta in sorted(shortlist,reverse=True):
        work.check()
        measured=[s.direct(pair,theta) for s in states]
        actual={key:sum(s.panel.objective_weight*m[key] for s,m in zip(states,measured)) for key in baseline}
        if feasible(actual["budget_w"],actual["budget_a"]) and actual["loss"]<best["loss"]:
            best=dict(actual,theta=theta)
    timings["verification"]=time.perf_counter()-t
    return {"status":"COMPLETE","baseline":baseline,"best":best,"accepted":best["theta"]!=0.,
            "event_count":len(angles),"event_incidences":work.generated,"candidate_count":candidate_count,
            "boundary_rounding_evaluations":boundary_count,"statistics":stats,
            "incremental_statistics":[o.update_stats for o in orbit],"timings":timings,
            "seconds":time.perf_counter()-start,"interval_certified":False,
            "candidate_policy":"32 lowest predicted feasible candidates; actual full-group verification",
            "scope":"FP64 numerical cell search, not certified/global optimal"}


def panel_fingerprint(panels):
    def digest(value):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()  # Artifact checksum only.
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return config_hash([{ "id":p.panel_id,"placement":p.placement,"tokens":p.token_ids_hash,
                         "weight":digest(p.weight),
                         "activation":digest(p.activation),
                         "shape":[list(p.weight.shape),list(p.activation.shape)],"objective_weight":p.objective_weight}
                       for p in panels])


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".rotation-",suffix=".npz",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f:
            arrays={k:v.detach().cpu().numpy() if hasattr(v,'detach') else v for k,v in arrays.items()}
            np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def panel_specs(specs, panel):
    """Resolve location-specific groups without changing any reduction axis."""
    selected=specs[panel.panel_id] if isinstance(specs,dict) else specs
    return selected if isinstance(selected,tuple) else (selected,selected)


def fit_shared_rotation(panels,q0,quant_specs,schedule,budgets=None,config=None,*,progress=None):
    start=time.perf_counter()
    config=config or SolverConfig()
    if config.backend=='torch':
        from .tensor_solver import fit_baseline as fit_tensor
        return fit_tensor('fixed_scale_search_matched' if config.fixed_scales else 'cellquant',
                          panels,q0,quant_specs,schedule,config,progress=progress,budgets=budgets)
    if config.backend!='numpy_reference':raise ValueError('Unknown solver backend')
    panels=list(panels);schedule=[tuple(map(int,p)) for p in schedule]
    if not panels or len({p.panel_id for p in panels})!=len(panels):
        raise ValueError("Nonempty panels with unique IDs required")
    q=np.asarray(q0,dtype=np.float64).copy();d=panels[0].hidden_size
    if q.shape!=(d,d) or any(p.hidden_size!=d for p in panels):raise ValueError("All panels must share one complete hidden dimension")
    report(progress,'solve.orthogonality',device='cpu')
    orth=float(np.linalg.norm(q.T@q-np.eye(d))/math.sqrt(d))
    if not np.isfinite(q).all() or orth>1e-6:raise ValueError("q0 is not numerically orthogonal")
    states=[]
    for p in panels:
        report(progress,'solve.initialize_panel',panel_id=p.panel_id,device=config.device)
        states.append(PanelState(p,q,*panel_specs(quant_specs,p),device=config.device,fixed=config.fixed_scales))
    initial=aggregate(states);limits=budgets or {k:initial[k] for k in ("budget_w","budget_a")}
    signature=config_hash({"panels":panel_fingerprint(panels),"q0":hashlib.sha256(q.tobytes()).hexdigest(),
                           "schedule":schedule,"specs":{s.panel.panel_id:[asdict(s.wspec),asdict(s.aspec)] for s in states},"limits":limits,
                           "config":{k:v for k,v in asdict(config).items() if k not in {"resume","checkpoint_path"}}})
    log=[];factors=[];offset=0;spent_before=0.
    if config.resume:
        if not config.checkpoint_path:raise ValueError("resume needs a checkpoint path")
        with np.load(config.checkpoint_path,allow_pickle=False) as ck:
            import json
            meta=json.loads(str(ck["metadata"]))
            if meta["signature"]!=signature:raise ValueError("Checkpoint config/source/environment/data mismatch; fork a new run")
            q=ck["q"].copy();log=meta["log"];factors=meta["factors"];offset=meta["next_pair"]
            spent_before=float(meta.get('elapsed_seconds',0.))
        # Reapply sparse factors to the same initialized panels, including the
        # original frozen scales, to avoid changing fixed-scale control semantics.
        for factor in factors:
            for state in states:state.accept(tuple(factor["pair"]),factor["theta"])
    def checkpoint(status,next_pair):
        if config.checkpoint_path:
            import json
            meta={"signature":signature,"status":status,"next_pair":next_pair,"log":log,"factors":factors,
                  "elapsed_seconds":spent_before+time.perf_counter()-start}
            atomic_npz(config.checkpoint_path,q=q,metadata=np.array(json.dumps(meta)))
    status="COMPLETE";reason=None
    next_pair=offset
    checkpoint('RUNNING',next_pair)
    try:
        for index in range(offset,len(schedule)):
            remaining=config.wall_seconds-spent_before-(time.perf_counter()-start)
            if remaining<=0:raise ResourceLimit("Total solver wall-clock budget exhausted")
            report(progress,'solve.pair',pair_index=index,pair_count=len(schedule),device=config.device)
            cfg=SolverConfig(**{**asdict(config),"seconds_per_pair":min(config.seconds_per_pair,remaining)})
            result=solve_pair(states,schedule[index],limits,cfg)
            result.update(pair=list(schedule[index]),pair_index=index)
            if result["accepted"]:
                theta=result["best"]["theta"]
                for state in states:state.accept(schedule[index],theta)
                rotate_rows(q,schedule[index],theta)
                factors.append({"pair":list(schedule[index]),"theta":theta})
            log.append(result);next_pair=index+1;checkpoint("RUNNING",next_pair)
    except ResourceLimit as exc:
        status="RESOURCE_BLOCKED";reason=str(exc);checkpoint(status,next_pair)
    except KeyboardInterrupt:
        # An interrupt can occur between panel updates, Q row writes and log
        # publication. Preserve the last complete atomic transaction instead
        # of serializing a partially updated Q with an older factor list.
        if config.checkpoint_path:
            import json
            with np.load(config.checkpoint_path,allow_pickle=False) as ck:
                safe_q=ck['q'].copy();meta=json.loads(str(ck['metadata']))
            meta.update(status='INTERRUPTED',elapsed_seconds=spent_before+time.perf_counter()-start)
            atomic_npz(config.checkpoint_path,q=safe_q,metadata=np.array(json.dumps(meta)))
        raise
    checkpoint(status,next_pair)
    return {"status":status,"reason":reason,"q":q,"factors":factors,"log":log,
            "initial":initial,"final":aggregate(states),"limits":limits,
            "orthogonality_residual":float(np.linalg.norm(q.T@q-np.eye(d))/math.sqrt(d)),
            "seconds":spent_before+time.perf_counter()-start,"signature":signature,
            "completed_pairs":next_pair,"requested_pairs":len(schedule),
            "shared_global_r1":True,"orientation":"R1=Q.T","interval_certified":False}
