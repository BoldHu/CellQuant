"""One disjoint Givens matching on a shared residual rotation."""
import math
import time

import numpy as np
import torch

from .events import ResourceLimit
from .progress import report
from .quantization import torch_quantize
from .solver import panel_specs
from .tensor_solver import PanelState, synchronize


def disjoint_pairs(dimension, seed):
    if dimension < 2 or dimension % 2:
        raise ValueError('A full matching needs an even residual dimension')
    rng = np.random.default_rng(np.random.SeedSequence([2026111802, seed]))
    return rng.permutation(dimension).reshape(-1, 2)


def proposal_grid(q, limit, count):
    if count < 3 or count % 2 != 1 or not 0 < limit < math.pi:
        raise ValueError('Expected an odd angle count and a positive finite angle interval below pi')
    half = count // 2
    # Zero first. Equal costs keep the smallest magnitude angle.
    steps = [0] + [j for i in range(1, half+1) for j in (-i, i)]
    return q.new_tensor(steps) * (limit / half)


def rotated_pairs(u, v, theta):
    cosine, sine = theta.cos(), theta.sin()
    return torch.stack((cosine*u-sine*v, sine*u+cosine*v), dim=1)


def apply_matching(q, pairs, angles):
    """Left-multiply by disjoint 2x2 rotations, preserving row order and full Q."""
    rows = torch.as_tensor(pairs, dtype=torch.long, device=q.device)
    if (rows.ndim != 2 or rows.shape[1] != 2 or len(rows) != len(angles) or not rows.numel()
            or int(rows.min()) < 0 or int(rows.max()) >= len(q) or rows.unique().numel() != rows.numel()):
        raise ValueError('Expected distinct, in-range axes and one angle per pair')
    theta = torch.as_tensor(angles, dtype=q.dtype, device=q.device).reshape(-1, 1)
    result = q.clone()
    result[rows.reshape(-1)] = rotated_pairs(q[rows[:,0]], q[rows[:,1]], theta).flatten(0, 1)
    return result


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise ResourceLimit('Output matching exceeded its calibration time budget')


def pair_costs(panels, q, specs, pairs, grid, criterion, *, batch_pairs=64, deadline=math.inf, progress=None):
    """Sum exact output-site contributions per plane and angle, tiling complete rows."""
    if criterion not in ('product','weight') or batch_pairs < 1:
        raise ValueError('Unknown matching cost or invalid tile size')
    rows = torch.as_tensor(pairs, device=q.device, dtype=torch.long)
    outputs = [p for p in panels if p.placement == 'output']
    if not outputs:
        raise ValueError('Output matching needs an output-side panel')
    costs = q.new_zeros((len(rows), len(grid)))
    output_risk = torch.zeros_like(costs)
    coverage = {}
    for panel in outputs:
        check_time(deadline)
        s = PanelState(panel, q, *panel_specs(specs, panel))
        row_energy = s.error.square().sum(dim=1)
        coverage[panel.panel_id] = (float(row_energy[rows].sum()/row_energy.sum())
                                    if bool(row_energy.sum() > 0) else None)
        report(progress, 'matching.cost_panel', panel=panel.panel_id, pair_count=len(rows), criterion=criterion)
        for start in range(0, len(rows), batch_pairs):
            check_time(deadline)
            tile = rows[start:start+batch_pairs]
            u, v = s.a[tile[:,0]], s.a[tile[:,1]]
            target_u, target_v = s.target[tile[:,0]], s.target[tile[:,1]]
            for k, theta in enumerate(grid):
                rotated = rotated_pairs(u, v, theta)
                decoded, _, scales = torch_quantize(rotated, s.wspec)
                if criterion == 'product':
                    target = rotated_pairs(target_u, target_v, theta)
                    error = (decoded.flatten(0,1) @ s.xh).reshape_as(target) - target
                    denominator = s.denominator
                else:
                    error = decoded - rotated
                    denominator = s.wnorm
                costs[start:start+len(tile), k] += panel.objective_weight * error.square().sum(dim=(1,2)) / denominator
                output_risk[start:start+len(tile), k] += (panel.objective_weight * s.wg / (4*s.wnorm)
                    * scales.square().reshape(len(tile),-1).sum(dim=1))
            # Bound the queued CUDA work as well as the host loop.
            synchronize(q.device)
        del s
    if not bool(torch.isfinite(costs).all()):
        raise ValueError('Nonfinite pair costs')
    return costs, coverage, output_risk-output_risk[:,:1]


def input_risk_bound(z, spec, norm, pairs, grid):
    """Per-plane upper increments of rowwise group AbsMax risk for any simultaneous matching."""
    width = spec.width(z.shape[1])
    peaks = z.reshape(len(z),-1,width).abs().amax(dim=-1).square().T
    left, right = pairs.T
    u, v = z.T[left], z.T[right]
    g, h = left//width, right//width
    max_u, max_v = peaks[g], peaks[h]
    same = (g == h)[:,None]
    bounds = z.new_empty((len(pairs), len(grid)))
    for k, theta in enumerate(grid):
        du = ((theta.cos()*u-theta.sin()*v).square()-max_u).clamp_min(0)
        dv = ((theta.sin()*u+theta.cos()*v).square()-max_v).clamp_min(0)
        excess = torch.where(same, torch.maximum(du,dv), du+dv)
        bounds[:,k] = excess.sum(dim=1) * (width/(4*spec.qmax**2)) / norm
    return bounds


def risk_tables(panels, q, specs, pairs, grid, output_delta, deadline):
    w = output_delta.clone()
    a = torch.zeros_like(w)
    indices = torch.as_tensor(pairs,dtype=torch.long,device=q.device)
    for panel in panels:
        if panel.placement == 'input':
            check_time(deadline)
            s = PanelState(panel,q,*panel_specs(specs,panel))
            w += panel.objective_weight * input_risk_bound(s.a,s.wspec,s.wnorm,indices,grid)
            a += panel.objective_weight * input_risk_bound(s.x.T,s.aspec,s.anorm,indices,grid)
            del s
    return {'budget_w':w,'budget_a':a}


def fit_totals(panels, q, specs):
    result = {'loss':0., 'output_loss':0., 'budget_w':0., 'budget_a':0.}
    for panel in panels:
        s = PanelState(panel, q, *panel_specs(specs, panel))
        weight = panel.objective_weight
        result['loss'] += weight * float(s.loss)
        if panel.placement == 'output':
            result['output_loss'] += weight * float(s.loss)
        result['budget_w'] += weight * float(s.bw)
        result['budget_a'] += weight * float(s.ba)
        del s
    if not all(map(math.isfinite, result.values())):
        raise ValueError('Nonfinite materialized fit metrics')
    return result


def fits_limits(values, limits, tolerance):
    return all(values[k] <= cap + tolerance * max(1., cap) for k, cap in limits.items())


@torch.inference_mode()
def fit_matching(panels, warm_q, specs, pairs, cfg, limits, *, criterion, grid_points=17,
                 step_scales=(0., .25, .5, 1.), allocate_risk=True, progress=None):
    """Propose from output sites. Accept only after full bilateral re-evaluation on fit documents."""
    synchronize(warm_q.device)
    start = time.monotonic()
    deadline = start + cfg.wall_seconds
    if not step_scales or list(step_scales) != sorted(set(step_scales)) or step_scales[0] != 0 or step_scales[-1] != 1:
        raise ValueError('Matching scales must increase from zero through one')
    initial = fit_totals(panels, warm_q, specs)
    if not fits_limits(initial, limits, cfg.budget_tolerance):
        raise ValueError('Materialized warm start violates the original fit risk limits')
    grid = proposal_grid(warm_q, cfg.theta_limit, grid_points)
    costs, coverage, output_delta = pair_costs(panels, warm_q, specs, pairs, grid, criterion, deadline=deadline, progress=progress)
    allocation = None
    if allocate_risk:
        bounds = risk_tables(panels,warm_q,specs,pairs,grid,output_delta,deadline)
        viable = torch.ones_like(costs,dtype=torch.bool)
        for key, bound in bounds.items():
            viable &= bound <= max(0.,limits[key]-initial[key])/len(pairs)
        if not bool(viable[:,0].all()):
            raise ValueError('Identity angle must satisfy both zero-increment bounds')
        chosen = costs.masked_fill(~viable,float('inf')).argmin(dim=1)
        allocation = {'feasible_angle_fraction':float(viable.double().mean()),
                      'full_step_risk_upper':{k:initial[k]+float(b.gather(1,chosen[:,None]).sum()) for k,b in bounds.items()}}
    else:
        chosen = costs.argmin(dim=1)
    angles = grid[chosen]
    proposed_change = float((costs.gather(1, chosen[:,None]).flatten() - costs[:,0]).sum())
    best_q, best, scale = warm_q, initial, 0.
    trials = [dict(scale=0., feasible=True, **initial)]
    for fraction in step_scales[1:]:
        check_time(deadline)
        report(progress, 'matching.joint_fit', scale=fraction, criterion=criterion)
        candidate_q = apply_matching(warm_q, pairs, angles * fraction)
        actual = fit_totals(panels, candidate_q, specs)
        feasible = fits_limits(actual, limits, cfg.budget_tolerance)
        trials.append(dict(scale=fraction, feasible=feasible, **actual))
        if feasible and actual['loss'] < best['loss']:
            best_q, best, scale = candidate_q, actual, fraction
    synchronize(warm_q.device)
    check_time(deadline)
    return best_q, {'initial':initial, 'final':best, 'limits':limits, 'criterion':criterion,
                    'grid_radians':grid.tolist(), 'pairs':np.asarray(pairs).tolist(), 'risk_allocation':allocation,
                    'proposed_angles':angles.tolist(), 'selected_scale':scale, 'joint_fit_trials':trials,
                    'output_residual_energy_covered':coverage, 'proposed_cost_change':proposed_change,
                    'product_replay_minus_predicted_change':(trials[-1]['output_loss'] - initial['output_loss'] - proposed_change
                                                              if criterion == 'product' else None),
                    'nonzero_applied_angles':int((angles*scale != 0).sum()), 'seconds':time.monotonic()-start}
