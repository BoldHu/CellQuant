"""Bounded FP64 event enumeration, including output-row envelopes.

These are numerical roots, not interval-certified event partitions. Input
orbits reduce the owner envelope to at most two moving members and the largest
unchanged member. Output orbits use the support polygon of all signed (b,c)
points, so they never apply the two-member screening lemma to whole rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
import numpy as np
from scipy.spatial import ConvexHull


class ResourceLimit(RuntimeError):
    """Incomplete search, never a scientific PASS or a failed method result."""


@dataclass
class WorkBudget:
    event_limit: int = 200000
    seconds: float = 300.0
    start: float = field(default_factory=time.perf_counter)
    generated: int = 0

    def check(self, extra=0):
        self.generated += extra
        if self.generated > self.event_limit:
            raise ResourceLimit(f"Event incidence budget exceeded: {self.generated}>{self.event_limit}")
        if time.perf_counter() - self.start > self.seconds:
            raise ResourceLimit(f"Pair wall-clock budget exceeded: {self.seconds:g}s")


def affine_roots(coefficients, left, right):
    """Vectorized a+b cos+c sin roots, without tolerance-merging narrow cells."""
    z = np.asarray(coefficients, dtype=np.float64).reshape(-1, 3)
    r = np.hypot(z[:, 1], z[:, 2])
    nz = r > 0
    ratio = np.divide(-z[:, 0], r, out=np.full_like(r, np.inf), where=nz)
    valid = nz & (np.abs(ratio) <= 1 + 8*np.finfo(float).eps)
    ids = np.flatnonzero(valid)
    phi = np.arctan2(z[ids, 2], z[ids, 1])
    alpha = np.arccos(np.clip(ratio[ids], -1, 1))
    roots, indices = [], []
    for sign in (-1, 1):
        base = phi + sign*alpha
        for k in (-1, 0, 1):
            t = base + k*2*math.pi
            residual = np.abs(z[ids, 0] + z[ids, 1]*np.cos(t) + z[ids, 2]*np.sin(t))
            ok = (t > left) & (t < right) & (residual <= 5e-13*np.maximum(np.abs(z[ids]).sum(1), 1e-300))
            roots.extend(t[ok].tolist())
            indices.extend(ids[ok].tolist())
    return roots, indices


def owner_intervals(coeff, left, right, stable=False):
    moving = np.any(coeff[:, 1:] != 0, axis=1)
    fixed = np.flatnonzero(~moving)
    variable = np.flatnonzero(moving)
    if len(fixed):
        fixed = fixed[[np.argmax(np.abs(coeff[fixed, 0]))]]
    if stable:
        owner = int(fixed[0])
        return [(left, right, owner, float(np.sign(coeff[owner, 0]))) ]
    candidates = [left, right]
    if len(variable) <= 2:
        members = np.r_[fixed, variable]
        for pos, i in enumerate(members):
            candidates.extend(affine_roots(coeff[i], left, right)[0])
            for j in members[pos+1:]:
                candidates.extend(affine_roots([coeff[i]-coeff[j], coeff[i]+coeff[j]], left, right)[0])
    else:
        if np.any(coeff[:, 0] != 0):
            raise ValueError("Whole-row orbits must have no constant term")
        points = np.unique(np.concatenate([coeff[:, 1:], -coeff[:, 1:]]), axis=0)
        # Degenerate support polygons (zero/collinear) have only sign switches.
        if len(points) < 3 or np.linalg.matrix_rank(points) < 2:
            owner = int(np.argmax(np.linalg.norm(coeff[:, 1:], axis=1)))
            candidates.extend(affine_roots(coeff[owner], left, right)[0])
        else:
            hull = ConvexHull(points)
            vertices = points[hull.vertices]
            edges = np.roll(vertices, -1, axis=0)-vertices
            roots, _ = affine_roots(np.c_[np.zeros(len(edges)), edges], left, right)
            candidates.extend(roots)
    candidates = np.unique(candidates)
    intervals = []
    for lo, hi in zip(candidates[:-1], candidates[1:]):
        theta = lo+(hi-lo)/2
        v = coeff @ np.array([1., math.cos(theta), math.sin(theta)])
        owner = int(np.argmax(np.abs(v)))
        signature = (owner, float(np.sign(v[owner])))
        if intervals and intervals[-1][2:] == signature:
            intervals[-1] = (intervals[-1][0], hi, *signature)
        else:
            intervals.append((lo, hi, *signature))
    return intervals


@dataclass
class GroupOrbit:
    side: str
    row: int
    start: int
    coeff: np.ndarray
    spec: object
    stable: bool = False
    fixed_scale: float | None = None

    def scale_coeff(self, theta):
        if self.fixed_scale is not None:
            return np.array([self.fixed_scale, 0., 0.])
        if self.stable:
            fixed = np.all(self.coeff[:, 1:] == 0, axis=1)
            return np.array([np.abs(self.coeff[fixed, 0]).max()/self.spec.qmax, 0., 0.])
        v = self.coeff @ np.array([1., math.cos(theta), math.sin(theta)])
        i = int(np.argmax(np.abs(v)))
        return np.sign(v[i])*self.coeff[i]/self.spec.qmax

    def reconstruction(self, theta, *, actual=False, indices=None):
        # Physical scale 0 with safe divisor 1 keeps zero groups continuous.
        basis = np.array([1., math.cos(theta), math.sin(theta)])
        sc = self.scale_coeff(theta)
        scale = max(float(sc @ basis), 0.)
        ix = slice(None) if indices is None else indices
        values = self.coeff[ix] @ basis
        code = np.clip(np.rint(values/(scale if scale > 0 else 1)), self.spec.qmin, self.spec.qmax)
        if actual:
            # Actual round-even at boundary, not a one-sided polynomial limit.
            return np.c_[code*scale, np.zeros((code.size, 2))], sc
        return code[:, None]*sc, sc

    def enumerate(self, left, right, screen, budget):
        moving = np.flatnonzero(np.any(self.coeff[:, 1:] != 0, axis=1))
        stable = self.stable and screen
        # Even fixed reconstruction scales need dynamic owner boundaries for
        # the common physical risk constraint.
        intervals = owner_intervals(self.coeff, left, right, stable)
        events = {}  # None = scale owner boundary; set = code members only.
        visits = 0
        for lo, hi, owner, sign in intervals:
            budget.check()
            for t in (lo, hi):
                events[t] = None
            scale = (np.array([self.fixed_scale, 0., 0.]) if self.fixed_scale is not None
                     else sign*self.coeff[owner]/self.spec.qmax)
            if self.stable:
                scale = self.scale_coeff((lo+hi)/2)
            ids = moving if stable or self.fixed_scale is not None else np.arange(len(self.coeff))
            visits += len(ids)
            # Chunk by threshold: bounded workspace even for full 11008-wide down.
            for threshold in np.arange(self.spec.qmin+.5, self.spec.qmax, 1):
                roots, which = affine_roots(self.coeff[ids]-threshold*scale, lo, hi)
                budget.check(len(roots))
                for theta, idx in zip(roots, which):
                    if theta not in events:
                        events[theta] = set()
                    if events[theta] is not None:
                        events[theta].add(int(ids[idx]))
        return events, {"owner_intervals": len(intervals), "member_visits": visits,
                        "stable_groups": int(self.stable), "groups": 1}
