"""Batched full-row input cells with sparse code-projection updates.

This path retains the entire reduction axis. Only rows/columns whose AbsMax
can change need new unrotated-member codes. Stable rows/columns keep their
cached contribution; the two rotating members and the bilateral cross term
are always included. Other grouping/dense-unstable cases use the existing
incremental solver, without dropping events or members.
"""
import torch


def stable_group_events(orbit, left, right, work):
    """Enumerate screened input groups together, retaining per-member incidences."""
    from .tensor_math import affine_roots
    by_block = {}
    for gid, group in enumerate(orbit.groups):
        if group.stable:
            by_block.setdefault((group.side, group.start), []).append(gid)
    events = {}
    state = orbit.state
    i, j = orbit.pair
    for (side, start), gids in by_block.items():
        values, scales = (state.a, state.sw) if side == 'w' else (state.x.T, state.sx)
        group = orbit.groups[gids[0]]
        width = len(group.coeff)
        moving = sorted(index for index in (i, j) if start <= index < start + width)
        member_ids = torch.tensor([index - start for index in moving], device=values.device)
        thresholds = torch.arange(group.spec.qmin + .5, group.spec.qmax, dtype=values.dtype, device=values.device)
        stats = {'owner_intervals': 1, 'member_visits': len(moving), 'stable_groups': 1, 'groups': 1}
        for offset in range(0, len(gids), 128):
            work.check()
            batch = gids[offset:offset+128]
            rows = torch.tensor([orbit.groups[gid].row for gid in batch], device=values.device)
            u, v = values[rows, i], values[rows, j]
            zero = torch.zeros_like(u)
            rotating = {i: torch.stack((zero, u, -v), -1), j: torch.stack((zero, v, u), -1)}
            coeff = torch.stack([rotating[index] for index in moving], 1)
            sc = torch.stack((scales[rows, start//width, 0], zero, zero), -1)
            equations = coeff[:, None] - thresholds[None, :, None, None] * sc[:, None, None, :]
            roots, eq = affine_roots(equations, left, right)
            work.check(len(roots))
            owner = eq // (len(thresholds) * len(moving))
            endpoints = values.new_tensor([left, right]).repeat(len(batch))
            points = torch.cat((endpoints, roots))
            members = torch.cat((eq.new_full((len(endpoints),), -1), member_ids[eq % len(moving)]))
            buckets = torch.cat((torch.arange(len(batch), device=values.device).repeat_interleave(2), owner))
            order = buckets.argsort(stable=True)
            # Only segment lengths leave the device. Sorting by group keeps
            # coincident angles and their distinct rounding incidences.
            lengths = buckets.bincount(minlength=len(batch)).tolist()
            for gid, a, ids in zip(batch, points[order].split(lengths), members[order].split(lengths)):
                events[gid] = a, ids, stats
    return events


def unstable_rows(orbit):
    return ([g.row for g in orbit.groups if g.side == 'w' and not g.stable],
            [g.row for g in orbit.groups if g.side == 'a' and not g.stable])


def can_batch(orbit):
    s = orbit.state
    if s.wg != s.a.shape[1] or s.ag != s.x.shape[0]:
        return False
    if s.fixed_w is not None or s.fixed_a is not None:
        return False
    uw, ux = unstable_rows(orbit)
    # An execution strategy, not a scientific filter: dense cases retain the
    # original exact incremental path. No unstable group is silently omitted.
    return 0 < len(uw) + len(ux) <= max(1, (s.a.shape[0] + s.x.shape[1]) // 4)


def _round(values, scale, spec):
    return (values / torch.where(scale > 0, scale, torch.ones_like(scale))).round().clamp(spec.qmin, spec.qmax)


class _Side:
    def __init__(self, values, pair, rows, spec):
        self.values, self.pair, self.spec = values, list(pair), spec
        self.rows = torch.tensor(rows, dtype=torch.long, device=values.device)
        self.moving = values[:, self.pair]
        self.rest = values.clone()
        self.rest[:, self.pair] = 0
        self.maximum = self.rest.abs().amax(-1)
        scale = values.abs().amax(-1, keepdim=True) / spec.qmax
        self.base = _round(self.rest, scale, spec)

    def sample(self, theta):
        u, v = self.moving.unbind(-1)
        c, s = theta.cos()[:, None], theta.sin()[:, None]
        rotating = torch.stack((c*u-s*v, s*u+c*v), -1)
        maxima = torch.cat((self.maximum.expand(len(theta), -1)[..., None], rotating.abs()), -1)
        maximum, owner = maxima.max(-1)
        scale = maximum / self.spec.qmax
        moving_codes = _round(rotating, scale[..., None], self.spec)
        # Decode every nonrotating member in every unstable group. This also
        # evaluates actual boundary rounding, including simultaneous events.
        codes = _round(self.rest[self.rows][None], scale[:, self.rows, None], self.spec)
        owner_coeff = torch.zeros((*scale.shape, 3), dtype=theta.dtype, device=theta.device)
        owner_coeff[..., 0] = torch.where(owner == 0, self.maximum, 0)
        sign0, sign1 = rotating[..., 0].sign(), rotating[..., 1].sign()
        owner_coeff[..., 1] = torch.where(owner == 1, sign0*u, torch.where(owner == 2, sign1*v, 0))
        owner_coeff[..., 2] = torch.where(owner == 1, -sign0*v, torch.where(owner == 2, sign1*u, 0))
        return scale, owner_coeff / self.spec.qmax, moving_codes, codes


def _prefix_projection(codes, opposite, stats):
    """C[b] @ opposite using one rebase and sparse code-difference prefix sums.

    Codes are integers represented exactly in FP64. Each batch is rebased;
    there is no full hidden-axis matrix product per cell. Batch order need
    not be monotone: differences telescope for boundaries and midpoints alike.
    """
    initial = codes[0] @ opposite
    delta = codes[1:] - codes[:-1]
    where = torch.nonzero(delta, as_tuple=False)
    updates = codes.new_zeros((len(codes), opposite.shape[1]))
    # Bound temporary gathers when many members change at the same boundary.
    for lo in range(0, len(where), 4096):
        idx = where[lo:lo+4096]
        updates.index_add_(0, idx[:, 0]+1,
                           delta[idx[:, 0], idx[:, 1], None] * opposite[idx[:, 1]])
    stats['input_projection_rebases'] = stats.get('input_projection_rebases', 0) + 1
    stats['input_code_projection_updates'] = stats.get('input_code_projection_updates', 0) + len(where)
    return initial + updates.cumsum(0)


def _square_quadratic(r):
    """Analytic squared degree-two residual -> degree-four loss coefficients."""
    a, b, c, d, e = r
    terms = (a*a+(b*b+c*c+d*d+e*e)/2,
             2*a*b+b*d+c*e, 2*a*c-c*d+b*e,
             2*a*d+(b*b-c*c)/2, 2*a*e+b*c,
             b*d-c*e, b*e+c*d, (d*d-e*e)/2, d*e)
    return torch.stack([v.sum((-2, -1)) for v in terms], -1)


def _risk_coeff(scales, factor):
    a, b, c = scales.unbind(-1)
    return torch.stack([v.sum(-1) for v in
                        (a*a+(b*b+c*c)/2, 2*a*b, 2*a*c, (b*b-c*c)/2, b*c)], -1)*factor


def full_row_input_term(orbit, angles, work, emit, batch_size):
    from .tensor_cells import CellTerm
    state = orbit.state
    uw, ux = unstable_rows(orbit)
    w = _Side(state.a, orbit.pair, uw, state.wspec)
    x = _Side(state.x.T, orbit.pair, ux, state.aspec)
    base = w.base @ x.base.T
    count = len(angles)-1
    m, d = state.a.shape
    tokens = state.x.shape[1]
    # Batch both boundaries and interior cells within a bounded workspace.
    elements = 2*(24*m*tokens + 6*d*(len(uw)+len(ux)) + 12*(m+tokens))
    batch = min(batch_size, max(1, (96*2**20)//(8*max(1, elements))))
    loss = angles.new_empty((count, 9))
    bw, ba = angles.new_empty((count, 5)), angles.new_empty((count, 5))
    boundary = angles.new_empty((count+1, 3))
    for lo in range(0, count+1, batch):
        work.check()
        hi = min(count+1, lo+batch)
        stop = min(hi, count)
        emit('batched_dynamic_input', panel_id=state.panel.panel_id,
             local_cell_index=lo, local_cell_count=count, output_row=None,
             unstable_weight_rows=len(uw), unstable_activation_columns=len(ux))
        theta = torch.cat((angles[lo:hi], angles[lo:stop]+(angles[lo+1:stop+1]-angles[lo:stop])/2))
        sw, wc, wp, wu = w.sample(theta)
        sx, xc, xp, xu = x.sample(theta)
        kernel = base.expand(len(theta), -1, -1).clone()
        for j, row in enumerate(uw):
            kernel[:, row] += _prefix_projection(wu[:, j], x.base.T, orbit.update_stats)-base[row]
        for j, col in enumerate(ux):
            kernel[:, :, col] += _prefix_projection(xu[:, j], w.base.T, orbit.update_stats)-base[:, col]
        if uw and ux:
            # The essential bilateral cross term; only the unstable
            # intersection contracts the full reduction axis at each point.
            cross = (wu-w.base[w.rows]) @ (xu-x.base[x.rows]).transpose(-1, -2)
            kernel[:, w.rows[:, None], x.rows[None, :]] += cross
        kernel += wp @ xp.transpose(-1, -2)
        n = hi-lo
        prediction = kernel[:n]*sw[:n, :, None]*sx[:n, None, :]
        boundary[lo:hi, 0] = (prediction-state.target).square().sum((-2, -1))/state.denominator
        boundary[lo:hi, 1] = sw[:n].square().sum(-1)*d/4/state.wnorm
        boundary[lo:hi, 2] = sx[:n].square().sum(-1)*d/4/state.anorm
        if stop > lo:
            a, b, c = wc[n:, :, None, :].unbind(-1)
            f, g, h = xc[n:, None, :, :].unbind(-1)
            k = kernel[n:]
            residual = (k*(a*f+(b*g+c*h)/2)-state.target,
                        k*(a*g+b*f), k*(a*h+c*f),
                        k*(b*g-c*h)/2, k*(b*h+c*g)/2)
            loss[lo:stop] = _square_quadratic(residual)/state.denominator
            bw[lo:stop] = _risk_coeff(wc[n:], d/4/state.wnorm)
            ba[lo:stop] = _risk_coeff(xc[n:], d/4/state.anorm)
    orbit.update_stats['input_batched_full_row_cells'] = count
    orbit.update_stats['input_unstable_weight_rows'] = len(uw)
    orbit.update_stats['input_unstable_activation_columns'] = len(ux)
    return CellTerm(angles, loss, bw, ba, boundary, state.panel.objective_weight)
