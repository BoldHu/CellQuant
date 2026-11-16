"""CPU reference for one bilateral, dynamically scaled Givens update.

This is a correctness oracle for SMALL frozen input-side panels, not the full
research method. It implements symmetric AbsMax RTN, full quantization groups,
dynamic scale-owner/code events, analytic cell coefficients, a lossless scale
screen, and numerical polynomial-root candidates with actual-quantizer checks.

Not implemented: a GPU/incremental event engine, output-side QW panels, full
Transformer calibration, GPTQ-in-the-objective, clipping/asymmetric quantizers,
or interval-arithmetic certification. Float64 event deduplication and root
filtering are numerical procedures, not proofs of a global minimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import platform
import time

import numpy as np
from numpy.polynomial import Polynomial


EVENT_TOL = 2e-12


def _unique(values, tol=EVENT_TOL):
    result = []
    for value in sorted(float(v) for v in values):
        if not result or value - result[-1] > tol:
            result.append(value)
    return np.asarray(result, dtype=np.float64)


def _affine_roots(coeff, left, right):
    """Interior roots of a+b*cos(theta)+c*sin(theta), in float64."""
    a, b, c = map(float, coeff)
    radius = math.hypot(b, c)
    scale = max(abs(a), radius, 1e-300)
    if radius <= 2e-14 * scale:
        return []  # Constant (including identically zero).
    target = -a / radius
    if abs(target) > 1 + 2e-13:
        return []
    phi = math.atan2(c, b)
    alpha = math.acos(float(np.clip(target, -1, 1)))
    result = []
    for base in (phi - alpha, phi + alpha):
        first = math.floor((left - base) / (2 * math.pi)) - 1
        last = math.ceil((right - base) / (2 * math.pi)) + 1
        for period in range(first, last + 1):
            theta = base + 2 * math.pi * period
            if left + EVENT_TOL < theta < right - EVENT_TOL:
                result.append(theta)
    return result


def givens(d, pair, theta):
    i, j = pair
    if i == j or not (0 <= i < d and 0 <= j < d):
        raise ValueError("A Givens pair needs two distinct in-range indices.")
    q = np.eye(d, dtype=np.float64)
    c, s = math.cos(theta), math.sin(theta)
    q[i, i] = q[j, j] = c
    q[i, j], q[j, i] = -s, s
    return q


def quantize_absmax(values, bits=4, group_size=-1):
    """Quantize LAST-axis complete groups; return float reconstruction/codes/scales.

    The representable signed code range includes -2**(b-1), as SpinQuant does.
    Without clipping, the AbsMax/positive-qmax scale leaves that lowest code
    unused in ordinary nonzero groups. np.rint implements ties-to-even.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Finite arrays with at least one axis are required.")
    if not 2 <= bits <= 8:
        raise ValueError("Reference bits must be between 2 and 8.")
    g = values.shape[-1] if group_size == -1 else int(group_size)
    if g <= 0 or values.shape[-1] % g:
        raise ValueError("group_size must divide the COMPLETE last dimension.")
    groups = values.reshape(*values.shape[:-1], -1, g)
    qmax, qmin = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))
    maximum = np.max(np.abs(groups), axis=-1, keepdims=True)
    scales = np.where(maximum > 0, maximum / qmax, 1.0)
    codes = np.clip(np.rint(groups / scales), qmin, qmax).astype(np.int64)
    return (codes * scales).reshape(values.shape), codes.reshape(values.shape), scales


@dataclass
class PairProblem:
    W: np.ndarray
    X: np.ndarray
    pair: tuple[int, int]
    bits: int = 4
    group_size: int = -1
    left: float = -math.pi / 4
    right: float = math.pi / 4

    def __post_init__(self):
        self.W = np.asarray(self.W, dtype=np.float64)
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.W.ndim != 2 or self.X.ndim != 2 or self.W.shape[1] != self.X.shape[0]:
            raise ValueError("Need W[m,d] and X[d,T].")
        if not np.all(np.isfinite(self.W)) or not np.all(np.isfinite(self.X)):
            raise ValueError("Nonfinite input.")
        self.m, self.d = self.W.shape
        self.t = self.X.shape[1]
        self.g = self.d if self.group_size == -1 else int(self.group_size)
        if self.g <= 0 or self.d % self.g:
            raise ValueError("Do not truncate groups: group_size must divide d.")
        if not (-math.pi < self.left < 0 < self.right < math.pi):
            raise ValueError("This reference uses an interval containing 0 inside (-pi,pi).")
        givens(self.d, self.pair, 0.0)
        if not 2 <= self.bits <= 8:
            raise ValueError("Reference bits must be in [2,8].")
        self.qmax = 2 ** (self.bits - 1) - 1
        self.qmin = -(2 ** (self.bits - 1))
        self.ng = self.d // self.g
        self.target = self.W @ self.X
        self.denominator = max(float(np.sum(self.target ** 2)), 1e-30)
        vectors = np.concatenate([self.W, self.X.T], axis=0)
        coeff = np.zeros((*vectors.shape, 3), dtype=np.float64)
        coeff[..., 0] = vectors
        i, j = self.pair
        coeff[:, [i, j], 0] = 0
        coeff[:, i, 1], coeff[:, i, 2] = vectors[:, i], -vectors[:, j]
        coeff[:, j, 1], coeff[:, j, 2] = vectors[:, j], vectors[:, i]
        self.coeff = coeff.reshape(-1, self.g, 3)
        radii = np.hypot(vectors[:, i], vectors[:, j])
        self.radii = np.repeat(radii, self.ng)
        self.changed = np.any(self.coeff[..., 1:] != 0, axis=-1)
        self.affected = np.any(self.changed, axis=1)
        maximum_other = np.max(
            np.where(self.changed, 0.0, np.abs(self.coeff[..., 0])), axis=1
        )
        self.maximum_other = maximum_other
        # Exact comparison, no positive tolerance: tolerance must not turn an
        # actually dynamic group into a falsely certified constant-scale group.
        self.stable = self.affected & (maximum_other >= self.radii)

    def values(self, theta):
        return self.coeff[..., 0] + self.coeff[..., 1] * np.cos(theta) + self.coeff[..., 2] * np.sin(theta)

    def state(self, theta):
        values = self.values(theta)
        owners = np.argmax(np.abs(values), axis=1)
        selected = values[np.arange(len(values)), owners]
        signs = np.sign(selected)
        maximum = np.abs(selected)
        scales = np.where(maximum > 0, maximum / self.qmax, 1.0)
        codes = np.clip(np.rint(values / scales[:, None]), self.qmin, self.qmax).astype(np.int64)
        return owners, signs, codes, scales

    def evaluate(self, theta):
        _, signs, codes, scales = self.state(theta)
        dequant = (codes * scales[:, None]).reshape(self.m + self.t, self.d)
        product = dequant[:self.m] @ dequant[self.m:].T
        loss = float(np.sum((product - self.target) ** 2) / self.denominator)
        split = self.m * self.ng
        physical_scales = np.where(signs == 0, 0., scales)
        bw = float(self.g * np.sum(physical_scales[:split] ** 2) / 4)
        ba = float(self.g * np.sum(physical_scales[split:] ** 2) / 4)
        return {"theta": float(theta), "loss": loss, "budget_w": bw, "budget_a": ba, "product": product}


def _group_owner_intervals(vector, left, right, force_stable=False):
    if force_stable:
        # A stable group's scale belongs to an UNCHANGED member, even when a
        # rotated member ties it at the interval midpoint. Re-selecting argmax
        # at that isolated tie would incorrectly use a sinusoidal scale.
        unchanged = np.all(vector[:, 1:] == 0, axis=1)
        owner = int(np.argmax(np.where(unchanged, np.abs(vector[:, 0]), -np.inf)))
        return [(left, right, owner, float(np.sign(vector[owner, 0])))]
    candidates = [left, right]
    for i in range(len(vector)):
        candidates.extend(_affine_roots(vector[i], left, right))
        for j in range(i + 1, len(vector)):
            candidates.extend(_affine_roots(vector[i] - vector[j], left, right))
            candidates.extend(_affine_roots(vector[i] + vector[j], left, right))
    candidates = _unique(candidates)
    kept, previous = [], None
    for lo, hi in zip(candidates[:-1], candidates[1:]):
        theta = (lo + hi) / 2
        values = vector[:, 0] + vector[:, 1] * np.cos(theta) + vector[:, 2] * np.sin(theta)
        owner = int(np.argmax(np.abs(values)))
        signature = (owner, float(np.sign(values[owner])))
        if signature != previous:
            kept.append([lo, hi, *signature])
        else:
            kept[-1][1] = hi
        previous = signature
    # Preserve the owner sampled in an OPEN subinterval. A midpoint of a
    # merged interval may be an isolated tie with a different argmax index.
    return [tuple(item) for item in kept]


def enumerate_events(problem, screen=True):
    """Full numerical event partition; safe screening does not remove loss terms."""
    if problem.d > 256 or problem.m + problem.t > 256:
        raise ValueError(
            "This CPU correctness oracle is limited to d<=256 and m+T<=256. "
            "Do not truncate real quantization groups to evade the limit; "
            "implement the production event backend before full-width LLM search."
        )
    events = [problem.left, problem.right]
    coordinates_enumerated = 0
    owner_cells = 0
    for row, vector in enumerate(problem.coeff):
        if not problem.affected[row]:
            continue
        stable = bool(screen and problem.stable[row])
        intervals = _group_owner_intervals(vector, problem.left, problem.right, stable)
        owner_cells += len(intervals)
        for lo, hi, owner, sign in intervals:
            events.extend([lo, hi])
            maximum_coeff = sign * vector[owner]
            if problem.stable[row]:
                # This remains valid with screening disabled: full enumeration
                # changes which members we visit, not the true scale formula.
                maximum_coeff = np.array([problem.maximum_other[row], 0.0, 0.0])
            if not np.any(maximum_coeff):
                scale_coeff = np.array([1.0, 0.0, 0.0])
            else:
                scale_coeff = maximum_coeff / problem.qmax
            indices = np.flatnonzero(problem.changed[row]) if stable else range(problem.g)
            for j in indices:
                coordinates_enumerated += 1
                for boundary in np.arange(problem.qmin + .5, problem.qmax, 1.):
                    events.extend(_affine_roots(vector[j] - boundary * scale_coeff, lo, hi))
    return _unique(events), {
        "screened": bool(screen),
        "affected_groups": int(problem.affected.sum()),
        "stable_affected_groups": int(problem.stable.sum()),
        "dynamic_affected_groups": int((problem.affected & ~problem.stable).sum()),
        "owner_cells": owner_cells,
        "member_owner_interval_visits": coordinates_enumerated,
    }


def _real_to_fourier(real):
    degree = (real.shape[-1] - 1) // 2
    result = np.empty((*real.shape[:-1], 2 * degree + 1), dtype=np.complex128)
    result[..., degree] = real[..., 0]
    for k in range(1, degree + 1):
        result[..., degree + k] = (real[..., 2*k-1] - 1j * real[..., 2*k]) / 2
        result[..., degree - k] = result[..., degree + k].conjugate()
    return result


def _fourier_to_real(coeff):
    degree = (len(coeff) - 1) // 2
    result = [float(coeff[degree].real)]
    for k in range(1, degree + 1):
        result.extend([float(2 * coeff[degree + k].real), float(-2 * coeff[degree + k].imag)])
    return np.asarray(result)


def trig_value(real, theta):
    result = np.asarray(real[..., 0]).copy()
    for k in range(1, (real.shape[-1] - 1) // 2 + 1):
        result = result + real[..., 2*k-1] * np.cos(k * theta) + real[..., 2*k] * np.sin(k * theta)
    return result


def analytic_cell_coefficients(problem, theta):
    owners, signs, codes, _ = problem.state(theta)
    scale_coeff = signs[:, None] * problem.coeff[np.arange(len(owners)), owners] / problem.qmax
    zero = signs == 0
    scale_coeff[zero] = [0, 0, 0]  # Safe divisor 1 is not a physical scale/risk.
    # At M=r, theta can be an isolated tie where np.argmax prefers a rotated
    # member. Its value is correct at theta but its function is not the scale
    # throughout the cell. Use the proven constant envelope for stable groups.
    scale_coeff[problem.stable] = np.stack([
        problem.maximum_other[problem.stable] / problem.qmax,
        np.zeros(np.count_nonzero(problem.stable)),
        np.zeros(np.count_nonzero(problem.stable)),
    ], axis=-1)
    dequant_coeff = (codes[..., None] * scale_coeff[:, None, :]).reshape(problem.m + problem.t, problem.d, 3)
    aw = [dequant_coeff[:problem.m, :, k] for k in range(3)]
    ax = [dequant_coeff[problem.m:, :, k].T for k in range(3)]
    p = np.stack([
        aw[0] @ ax[0] + (aw[1] @ ax[1] + aw[2] @ ax[2]) / 2,
        aw[0] @ ax[1] + aw[1] @ ax[0],
        aw[0] @ ax[2] + aw[2] @ ax[0],
        (aw[1] @ ax[1] - aw[2] @ ax[2]) / 2,
        (aw[1] @ ax[2] + aw[2] @ ax[1]) / 2,
    ], axis=-1)
    error = _real_to_fourier(p)
    error[..., 2] -= problem.target
    loss_fourier = np.zeros(9, dtype=np.complex128)
    for residual in error.reshape(-1, 5):
        loss_fourier += np.convolve(residual, residual)
    loss_real = _fourier_to_real(loss_fourier / problem.denominator)
    a, b, c = scale_coeff.T
    budget_by_group = np.stack([a*a+(b*b+c*c)/2, 2*a*b, 2*a*c, (b*b-c*c)/2, b*c], axis=-1) * problem.g / 4
    split = problem.m * problem.ng
    return p, loss_real, budget_by_group[:split].sum(0), budget_by_group[split:].sum(0)


def _trig_polynomial(real):
    """(1+t^2)^degree f(2 atan(t)); coefficients in increasing powers."""
    degree = (len(real) - 1) // 2
    den = Polynomial([1., 0., 1.])
    result = real[0] * den ** degree
    for k in range(1, degree + 1):
        power = (Polynomial([1., 1j]) ** (2*k)).coef
        numerator = Polynomial(real[2*k-1] * power.real + real[2*k] * power.imag)
        result = result + numerator * den ** (degree - k)
    coeff = np.asarray(result.coef, dtype=np.float64)
    scale = float(np.max(np.abs(coeff))) if len(coeff) else 0.0
    if scale == 0:
        return np.zeros(1)
    while len(coeff) > 1 and abs(coeff[-1]) < 1e-13 * scale:
        coeff = coeff[:-1]
    return coeff / scale


def _trig_roots(real, left, right):
    coeff = _trig_polynomial(real)
    if len(coeff) <= 1:
        return []
    roots = Polynomial(coeff).roots()
    result = []
    scale = max(float(np.sum(np.abs(real))), 1e-30)
    for root in roots:
        if abs(root.imag) <= 2e-7 * max(1., abs(root.real)):
            theta = 2 * math.atan(float(root.real))
            if left + EVENT_TOL < theta < right - EVENT_TOL:
                if abs(float(trig_value(real, theta))) < 2e-7 * scale:
                    result.append(theta)
    return _unique(result).tolist()


def solve_pair_reference(problem, screen=True, constrain_budgets=True):
    """Numerical analytic-cell candidate solver, not a certified optimizer.

    Actual boundary values, nearby one-sided values, feasible cell midpoints,
    stationary candidates and theta=0 are checked by the actual quantizer.
    The only unconditional numerical acceptance claim is no accepted increase
    in this frozen-panel loss/budgets within the declared float64 tolerances.
    """
    start = time.perf_counter()
    baseline = problem.evaluate(0.)
    events, statistics = enumerate_events(problem, screen)
    candidates = [0., *events.tolist()]
    for left, right in zip(events[:-1], events[1:]):
        _, loss, bw, ba = analytic_cell_coefficients(problem, (left + right) / 2)
        derivative = np.zeros_like(loss)
        for k in range(1, 5):
            derivative[2*k-1] = k * loss[2*k]
            derivative[2*k] = -k * loss[2*k-1]
        stationary = _trig_roots(derivative, left, right)
        feasible_events = [left, right]
        if constrain_budgets:
            for budget, limit in ((bw, baseline['budget_w']), (ba, baseline['budget_a'])):
                boundary = budget.copy()
                boundary[0] -= limit
                feasible_events.extend(_trig_roots(boundary, left, right))
        feasible_events = _unique(feasible_events)
        candidates.extend(feasible_events)
        for lo, hi in zip(feasible_events[:-1], feasible_events[1:]):
            middle = (lo + hi) / 2
            if constrain_budgets and (
                trig_value(bw, middle) > baseline['budget_w'] + 1e-10 * max(1., baseline['budget_w'])
                or trig_value(ba, middle) > baseline['budget_a'] + 1e-10 * max(1., baseline['budget_a'])
            ):
                continue
            candidates.append(middle)
            # Both representable neighbors and safely interior one-sided probes.
            candidates.extend([np.nextafter(lo, hi), np.nextafter(hi, lo), lo+(hi-lo)*1e-7, hi-(hi-lo)*1e-7])
            candidates.extend(t for t in stationary if lo < t < hi)
    # Do not deduplicate with EVENT_TOL here: near-boundary candidates can have
    # different round-to-even codes and must be evaluated separately.
    candidates = np.unique(np.asarray(candidates, dtype=np.float64))
    best = baseline
    feasible_count = 0
    for theta in candidates:
        value = problem.evaluate(float(theta))
        if constrain_budgets and (
            value['budget_w'] > baseline['budget_w'] + 1e-10 * max(1., baseline['budget_w'])
            or value['budget_a'] > baseline['budget_a'] + 1e-10 * max(1., baseline['budget_a'])
        ):
            continue
        feasible_count += 1
        if value['loss'] < best['loss']:
            best = value
    return {
        "baseline": {k: v for k, v in baseline.items() if k != 'product'},
        "best": {k: v for k, v in best.items() if k != 'product'},
        "accepted": bool(best['theta'] != 0. and best['loss'] < baseline['loss']),
        "event_count": int(len(events)), "open_cell_count": int(len(events)-1),
        "candidate_count": int(len(candidates)), "feasible_candidate_count": feasible_count,
        "statistics": statistics, "seconds": time.perf_counter()-start,
        "interval_certified": False, "global_Od_optimum": False,
        "budget_tolerance": "1e-10 * max(1, initial_budget)",
    }


def run_reference_checks(output_dir, seed=20261115):
    """Run real small-matrix checks and write math_reference_report.json."""
    start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    W, X = rng.normal(size=(8, 16)), rng.normal(size=(16, 6))
    problem = PairProblem(W, X, (5, 13))
    full_events, full_stats = enumerate_events(problem, False)
    screened_events, screened_stats = enumerate_events(problem, True)
    events_equal = len(full_events) == len(screened_events) and np.allclose(full_events, screened_events, rtol=0, atol=1e-10)
    orthogonal_error, product_error = 0., 0.
    for theta in rng.uniform(-1, 1, 12):
        q = givens(16, problem.pair, theta)
        orthogonal_error = max(orthogonal_error, float(np.linalg.norm(q.T @ q - np.eye(16))))
        product_error = max(product_error, float(np.linalg.norm((W @ q.T) @ (q @ X)-W@X) / np.linalg.norm(W@X)))
    unstable = 0
    max_product_err, max_loss_err, max_budget_err = 0., 0., 0.
    for left, right in zip(screened_events[:-1], screened_events[1:]):
        p, loss, bw, ba = analytic_cell_coefficients(problem, (left+right)/2)
        signatures = []
        for frac in (.17, .43, .79):
            theta = left + frac*(right-left)
            owner, sign, codes, _ = problem.state(theta)
            signatures.append((owner.tobytes(), sign.tobytes(), codes.tobytes()))
            actual = problem.evaluate(theta)
            max_product_err = max(max_product_err, float(np.linalg.norm(trig_value(p, theta)-actual['product']) / max(np.linalg.norm(actual['product']), 1e-30)))
            max_loss_err = max(max_loss_err, abs(float(trig_value(loss, theta))-actual['loss']))
            max_budget_err = max(max_budget_err, abs(float(trig_value(bw, theta))-actual['budget_w']), abs(float(trig_value(ba, theta))-actual['budget_a']))
        unstable += int(len(set(signatures)) != 1)
    solution = solve_pair_reference(problem, True, True)
    before, after = solution['baseline'], solution['best']
    q_best = givens(16, problem.pair, after['theta'])
    independent_w = quantize_absmax(W @ q_best.T)[0]
    independent_x = quantize_absmax((q_best @ X).T)[0].T
    independent_loss = float(np.sum((independent_w @ independent_x-W@X)**2) / problem.denominator)
    # Exact ties and all-zero-group behavior are tested without analytic events.
    tie_values = np.array([[7., .5, 1.5, 2.5, -.5, -1.5, -2.5, -7.]])
    _, tie_codes, _ = quantize_absmax(tie_values)
    zero_hat, zero_codes, zero_scales = quantize_absmax(np.zeros((2,16)), group_size=8)
    metrics = {
        "fp64_orthogonality_fro": orthogonal_error,
        "fp64_product_relative_l2": product_error,
        "screened_vs_full_events_equal": bool(events_equal),
        "sample_detected_unstable_cells": unstable,
        "max_product_degree2_relative_l2": max_product_err,
        "max_loss_degree4_absolute_error": max_loss_err,
        "max_budget_degree2_absolute_error": max_budget_err,
        "actual_loss_nonincreasing": bool(after['loss'] <= before['loss'] + 1e-12),
        "actual_budget_w_feasible": bool(after['budget_w'] <= before['budget_w']+1e-9*max(1., before['budget_w'])),
        "actual_budget_a_feasible": bool(after['budget_a'] <= before['budget_a']+1e-9*max(1., before['budget_a'])),
        "independent_requantized_loss_absolute_error": abs(independent_loss-after['loss']),
        "ties_to_even_correct": bool(np.array_equal(tie_codes, [[7,0,2,2,0,-2,-2,-7]])),
        "all_zero_groups_finite_and_exact": bool(np.all(np.isfinite(zero_scales)) and not np.any(zero_hat) and not np.any(zero_codes)),
    }
    checks = {
        "orthogonality": orthogonal_error < 1e-12,
        "product_equivalence": product_error < 1e-12,
        "screening_event_equivalence": bool(events_equal),
        "cell_partition": unstable == 0,
        "analytic_product": max_product_err < 1e-10,
        "analytic_loss": max_loss_err < 1e-10,
        "analytic_budgets": max_budget_err < 1e-9,
        "actual_acceptance": all(metrics[k] for k in ('actual_loss_nonincreasing','actual_budget_w_feasible','actual_budget_a_feasible')),
        "independent_requantization": metrics['independent_requantized_loss_absolute_error'] < 1e-12,
        "rounding_and_zero": metrics['ties_to_even_correct'] and metrics['all_zero_groups_finite_and_exact'],
    }
    result = {
        "schema_version": 1, "status": "PASS" if all(checks.values()) else "FAIL",
        "seed": seed, "evidence_kind": "EXECUTED_CPU_SYNTHETIC_REFERENCE", "llm_experiment": False,
        "source_type": "synthetic_math", "scope": "input_side_single_pair", "global_R1_model_implemented": False,
        "W_shape": list(W.shape), "X_shape": list(X.shape), "pair": list(problem.pair),
        "quantizer": {"bits":4,"codes":[-8,7],"group_size":16,"symmetric":True,"clip_ratio":1.,"rounding":"ties_to_even"},
        "checks": checks, "metrics": metrics,
        "full_event_statistics": full_stats, "screened_event_statistics": screened_stats,
        "solution": solution, "seconds_total": time.perf_counter()-start,
        "runtime": {"python":platform.python_version(),"numpy":np.__version__,"device":"CPU"},
        "implemented_scope": ["complete input-axis groups", "symmetric dynamic AbsMax RTN", "one bilateral Givens coordinate", "numerical full event partition", "analytic degree-2 product / degree-4 loss", "safe scale screening", "numerical stationary/budget-root candidates", "actual boundary/near-boundary evaluation", "actual loss/budget acceptance"],
        "limits": ["NOT an LLM result or migration gate by itself", "NOT interval-arithmetic certified", "NOT a global optimum over O(d)", "NOT a GPU/incremental production solver", "reference event search restricted to d<=256 and m+T<=256; do not truncate real groups", "output-side QW objective NOT_IMPLEMENTED", "clipping/asymmetric/GPTQ objectives NOT_IMPLEMENTED", "frozen-panel guarantees do not imply held-out or PPL improvement", "polynomial root filtering and event deduplication have float64 tolerances"],
    }
    path = output_dir / 'math_reference_report.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    return result
