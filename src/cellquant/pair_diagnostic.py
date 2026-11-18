"""Fit-only comparison of two coordinate schedules at the same shared Hadamard Q."""
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch

from .events import ResourceLimit
from .pilot import pilot_method_budget
from .progress import report
from .runtime import atomic_json, config_hash
from .solver import SolverConfig, pair_schedule, panel_fingerprint, panel_specs
from .tensor_solver import (MATCHED_GRID_POINTS, PanelState, aggregate, evaluate_proposals,
                            grid_angles, hadamard_rotation, numerical_context, plain,
                            rotate_rows, solve_pair, synchronize)


PROBE_COUNT = 8


def compare_support(states, schedule):
    """Relabel vertices only. Preserve pair order, repetitions and orientations."""
    axes = sorted({i for pair in schedule for i in pair})
    outputs = [s for s in states if s.panel.placement == 'output']
    if not outputs or not axes:
        raise ValueError('Expected output-side panels and a nonempty coordinate schedule')
    rows = torch.stack([s.panel.objective_weight * s.error.square().sum(dim=1) / s.denominator
                        for s in outputs])
    scores = rows.sum(dim=0)
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite output residual energy')
    # One ranking for all output panels. Stable sort breaks ties by coordinate ID.
    order = torch.argsort(scores, descending=True, stable=True).tolist()
    support = set(axes)
    old_rank = [i for i in order if i in support]
    if len(old_rank) != len(axes) or any(i == j for i, j in schedule):
        raise ValueError('Invalid rotation coordinates')
    selected = order[:len(axes)]
    mapping = dict(zip(old_rank, selected))
    ranked = [(mapping[i], mapping[j]) for i, j in schedule]

    def coverage(energy):
        total = float(energy.sum())
        return {arm: float(energy[ids].sum()) / total if total > 0 else None
                for arm, ids in [('original', axes), ('ranked', selected)]}

    return ranked, {'distinct_axes': len(axes), 'hidden_size': len(order),
                    'output_objective_bound_fraction': coverage(scores),
                    'per_output_bound_fraction': {s.panel.panel_id: coverage(row)
                                                  for s, row in zip(outputs, rows)},
                    'original_axes': axes, 'ranked_axes': selected}


def replay_angle(states, specs, q0, pair, theta):
    """Materialize G @ Q0 and requantize each full panel, as in held-out evaluation."""
    q = q0.clone()
    rotate_rows(q, pair, theta)
    replayed = {'loss': 0., 'budget_w': 0., 'budget_a': 0.}
    by_side = {'input': 0., 'output': 0.}
    for initial in states:
        panel = initial.panel
        current = PanelState(panel, q, *panel_specs(specs, panel))
        delta = float(initial.loss - current.loss)
        by_side[panel.placement] += panel.objective_weight * delta
        metrics = {'loss': current.loss, 'budget_w': current.bw, 'budget_a': current.ba}
        for key, value in metrics.items():
            replayed[key] += panel.objective_weight * float(value)
        del current
    return replayed, by_side


@torch.inference_mode()
def diagnose_seed(panels, specs, q0, schedule, config, *, seed, output_path, progress=None):
    """Measure actual feasible reductions without applying or accumulating any probe."""
    if q0.dtype != torch.float64 or config.compute_dtype != 'float64' or not config.constrain_budgets:
        raise ValueError('Fit diagnostic requires the constrained FP64 protocol')
    sources = {p.metadata.get('source_type') for p in panels}
    if len(sources) != 1 or not sources <= {'real_layer', 'unit_test_fixture', 'synthetic_math'}:
        raise ValueError('Expected consistent measured panel provenance')
    numerical_context(q0.device)
    synchronize(q0.device)
    start = time.perf_counter()
    states = [PanelState(p, q0, *panel_specs(specs, p)) for p in panels]
    initial = aggregate(states)
    limits = {k: initial[k] for k in ('budget_w', 'budget_a')}
    synchronize(q0.device)
    init_seconds = time.perf_counter() - start
    start = time.perf_counter()
    ranked, coverage = compare_support(states, schedule)
    synchronize(q0.device)
    ranking_seconds = time.perf_counter() - start
    schedules = {'original': schedule, 'ranked': ranked}
    positions = np.linspace(0, len(schedule) - 1, min(PROBE_COUNT, len(schedule)), dtype=int).tolist()
    baseline = float(initial['loss'])
    result = {'status': 'IN_PROGRESS', 'evidence_kind': 'FIT_PAIR_DIAGNOSTIC', 'source_type': sources.pop(),
              'seed': seed, 'fit_panel_sha256': panel_fingerprint(panels), 'panel_ids': [p.panel_id for p in panels],
              'panel_token_hashes': {p.panel_id: p.token_ids_hash for p in panels}, 'device': str(q0.device),
              'initial': plain(initial), 'coverage': coverage, 'initialization_seconds': init_seconds,
              'ranking_seconds': ranking_seconds, 'pair_budget': len(schedule), 'grid_points': MATCHED_GRID_POINTS,
              'theta_limit': config.theta_limit, 'event_limit_per_pair': config.event_limit_per_pair,
              'seconds_per_pair': config.seconds_per_pair, 'method_wall_seconds': config.wall_seconds,
              'schedule_sha256': {arm: config_hash(pairs) for arm, pairs in schedules.items()},
              'probe_positions_zero_based': positions, 'probes': [],
              'scope': 'Independent single-pair probes at Q0 on fit only. Output-row error mass bounds '
                       'possible output-side recovery on a fixed support. Probes are not additive.'}
    atomic_json(output_path, result)
    spent = {(arm, method): init_seconds + (ranking_seconds if arm == 'ranked' else 0.)
             for arm in schedules for method in ('grid', 'cellquant')}
    last_probe = None
    try:
        for order, index in enumerate(positions):
            arms = ('original', 'ranked') if order % 2 == 0 else ('ranked', 'original')
            for arm_idx, arm in enumerate(arms):
                pair = schedules[arm][index]
                methods = ('grid', 'cellquant') if (order + arm_idx) % 2 == 0 else ('cellquant', 'grid')
                for method in methods:
                    last_probe = {'position': index, 'arm': arm, 'method': method}
                    report(progress, 'diagnostic.probe', seed=seed, **last_probe)
                    remaining = config.wall_seconds - spent[arm, method]
                    if remaining <= 0:
                        raise ResourceLimit('Diagnostic method wall budget exhausted')
                    cap = min(config.seconds_per_pair, remaining)
                    synchronize(q0.device)
                    start = time.perf_counter()
                    if method == 'grid':
                        angles = grid_angles(q0, config.theta_limit)
                        row = evaluate_proposals(states, pair, limits, config, angles, start + cap)
                        if row['evaluations'] != len(angles):
                            raise ResourceLimit('Diagnostic matched grid did not finish every angle')
                    else:
                        row = solve_pair(states, pair, limits, replace(config, seconds_per_pair=cap))
                    synchronize(q0.device)
                    elapsed = time.perf_counter() - start
                    spent[arm, method] += elapsed
                    if elapsed > cap:
                        raise ResourceLimit('Diagnostic pair wall budget exhausted')
                    best = row['best']
                    start = time.perf_counter()
                    replayed, by_side = replay_angle(states, specs, q0, pair, best['theta'])
                    feasible = all(replayed[k] <= float(bound) + config.budget_tolerance * max(1., float(bound))
                                   for k, bound in limits.items())
                    replay_gain = 100 * (baseline - replayed['loss']) / baseline if baseline else None
                    feasible_gain = max(0., replay_gain) if replay_gain is not None else None
                    if not feasible and baseline:
                        feasible_gain = 0.
                    synchronize(q0.device)
                    result['probes'].append({**last_probe, 'pair': list(pair), 'best': best, 'seconds': elapsed,
                        'replayed': replayed, 'replayed_joint_loss_drop_by_side': by_side,
                        'replay_seconds': time.perf_counter() - start, 'replayed_feasible': feasible,
                        'feasible_replay_gain_pct': feasible_gain,
                        'grid_evaluations': row.get('evaluations'),
                        'event_incidences': row.get('event_incidences'), 'event_candidates': row.get('candidate_count'),
                        'gain_pct_of_initial_joint_loss': 100 * (baseline - best['loss']) / baseline if baseline else None})
                    atomic_json(output_path, result)
    except (ResourceLimit, torch.cuda.OutOfMemoryError, KeyboardInterrupt) as exc:
        result.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'RESOURCE_BLOCKED',
                      reason=str(exc), last_probe=last_probe)
        atomic_json(output_path, result)
        raise
    def summaries(field):
        rows = {}
        if baseline > 0:
            for arm in schedules:
                means = {m: float(np.mean([r[field] for r in result['probes'] if r['arm'] == arm
                                           and r['method'] == m])) for m in ('grid', 'cellquant')}
                rows[arm] = dict(means, cell_minus_grid_pp=means['cellquant'] - means['grid'])
        return rows
    result.update(status='COMPLETE', mean_single_pair_gain_pct=summaries('gain_pct_of_initial_joint_loss'),
                  mean_feasible_replay_gain_pct=summaries('feasible_replay_gain_pct'))
    atomic_json(output_path, result)
    return result


def run_pair_diagnostic(config, prepared, output, *, progress=None):
    settings = config['solver']
    solver = SolverConfig(backend=settings['backend'], device=settings['device'], compute_dtype=settings['compute_dtype'],
                          theta_limit=settings.get('theta_limit', np.pi/4), cell_batch_size=settings.get('cell_batch_size', 1024),
                          event_limit_per_pair=settings.get('event_limit_per_pair', 200000),
                          seconds_per_pair=settings.get('seconds_per_pair', 300.), wall_seconds=pilot_method_budget(config))
    paths, summaries, sources = [], {}, set()
    for seed in config['seeds']:
        ready = prepared['seeds'].pop(seed)
        if 'heldout' in ready:
            raise ValueError('Expected fit-only capture')
        q0 = hadamard_rotation(prepared['hidden_size'], seed, solver.device)
        pairs = pair_schedule(len(q0), settings['pair_budget'], seed)
        path = Path(output) / f'seed_{seed}' / 'pair_diagnostic.json'
        row = diagnose_seed(ready['fit'], ready['specs'], q0, pairs, solver, seed=seed, output_path=path, progress=progress)
        summaries[str(seed)] = {'capture_seconds': ready['capture_seconds'], 'ranking_seconds': row['ranking_seconds'],
                                'coverage': row['coverage']['output_objective_bound_fraction'],
                                'mean_single_pair_gain_pct': row['mean_single_pair_gain_pct'],
                                'mean_feasible_replay_gain_pct': row['mean_feasible_replay_gain_pct']}
        sources.add(row['source_type'])
        paths.append(str(path))
        del ready, q0
    if len(sources) != 1:
        raise ValueError('Inconsistent cross-seed panel provenance')
    return {'source_type': sources.pop(), 'status': 'DIAGNOSTIC_COMPLETE', 'evidence_kind': 'FIT_PAIR_DIAGNOSTIC',
            'diagnostic_results': paths, 'paired_fit_summary': summaries}
