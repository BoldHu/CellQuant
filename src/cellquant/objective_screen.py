"""Prospective four-arm comparison on a document split within the original G1 fit set."""
from dataclasses import replace
from pathlib import Path

import numpy as np

from .baselines import fit_baseline
from .data import load_token_manifest
from .pilot import panel_metrics, pilot_solver_config
from .progress import report
from .runtime import atomic_json, config_hash, read_json, sha256_file
from .solver import pair_schedule, panel_fingerprint
from .tensor_solver import hadamard_rotation


SCREEN_PAIRS = 32
MIN_GRID_GAIN_PCT = 1.0
OLD, NEW = 'sum_nmse', 'relative_site_nmse'
GRID, CELL, HAD = 'angle_grid_matched', 'cellquant', 'hadamard_shared'
ARMS = ((OLD, GRID), (NEW, GRID), (OLD, CELL), (NEW, CELL))


def validate_screen_config(config):
    formal, solver = config.get('formal_pilot', {}), config.get('solver', {})
    expected = {'fit_sequences': 8, 'heldout_sequences': 8, 'fit_tokens_per_panel': 128,
                'heldout_tokens_per_panel': 256, 'context_length': 2048, 'input_weight_rows': 64}
    if (config.get('seeds') != [11, 23] or any(formal.get(k) != v for k, v in expected.items())
            or formal.get('layers') != [0, 15, 31] or formal.get('projections') != ['q_proj','up_proj','down_proj']
            or solver.get('pair_budget') != 128 or solver.get('fit_objective', OLD) != OLD):
        raise ValueError('Objective screen requires the original two-seed, nine-site, 128-pair G1 protocol')


def verify_parent_manifests(config):
    protocol = Path(config.get('protocol_manifest_path') or '.')
    if not protocol.is_file() or sha256_file(protocol) != config.get('protocol_sha256'):
        raise ValueError('Original frozen G1 protocol is missing or changed')
    frozen = read_json(protocol).get('token_manifest_sha256', {})
    manifests = config.get('data', {}).get('token_manifests', {})
    if (set(frozen) != {str(s) for s in config['seeds']} or set(manifests) != set(frozen)
            or any(not Path(p).is_file() or sha256_file(p) != frozen[s] for s, p in manifests.items())):
        raise ValueError('Original G1 token manifests disagree with the frozen protocol')


def assess_screen(rows):
    """Ratios within seed/site, then mean over seeds and median over sites, as for G1."""
    seeds = sorted({row['seed'] for row in rows})
    sites = sorted(p['panel_id'] for p in rows[0]['validation']) if rows else []
    if not seeds or not sites or len(set(sites)) != len(sites):
        raise ValueError('Expected nonempty distinct validation sites')
    values = {}
    for row in rows:
        panels = {p['panel_id']: p['nmse'] for p in row['validation']}
        key = row['seed'], row['objective'], row['method']
        if key in values or set(panels) != set(sites) or len(panels) != len(row['validation']):
            raise ValueError('Unpaired or duplicate objective screen result')
        values[key] = [panels[s] for s in sites]
    arms = ((OLD, HAD), *ARMS)
    if set(values) != {(s, *arm) for s in seeds for arm in arms}:
        raise ValueError('Missing matched objective screen arms')
    tables = {arm: np.array([values[(s, *arm)] for s in seeds]) for arm in arms}
    if any(not np.isfinite(t).all() or (t < 0).any() for t in tables.values()):
        raise ValueError('Invalid validation NMSE')
    if any((tables[arm] == 0).any() for arm in ((OLD,HAD), (OLD,GRID), (NEW,GRID), (OLD,CELL))):
        return {'decision': 'INSUFFICIENT_EVIDENCE', 'reason': 'A comparator has zero validation NMSE'}
    candidate = tables[NEW, CELL]
    def contrast(new, other):
        gain = 100 * (1 - new / other)
        return {'median_site_gain_pct': float(np.median(gain.mean(axis=0))),
                'seed_median_gain_pct': {str(s): float(np.median(g)) for s, g in zip(seeds, gain)},
                'site_gain_pct': dict(zip(sites, gain.mean(axis=0).tolist()))}
    comparisons = {
        'relative_cell_vs_stronger_grid': contrast(candidate, np.minimum(tables[OLD,GRID], tables[NEW,GRID])),
        'relative_cell_vs_original_cell': contrast(candidate, tables[OLD,CELL]),
        'relative_cell_vs_relative_grid': contrast(candidate, tables[NEW,GRID]),
        'original_cell_vs_original_grid': contrast(tables[OLD,CELL], tables[OLD,GRID]),
        'relative_cell_vs_hadamard': contrast(candidate, tables[OLD,HAD])}
    strongest = comparisons['relative_cell_vs_stronger_grid']
    promote = (strongest['median_site_gain_pct'] >= MIN_GRID_GAIN_PCT
               and min(strongest['seed_median_gain_pct'].values()) > 0
               and comparisons['relative_cell_vs_original_cell']['median_site_gain_pct'] > 0)
    return {'decision': 'TEST_ONE_G1' if promote else 'STOP', 'comparisons': comparisons,
            'minimum_screen_gain_pct': MIN_GRID_GAIN_PCT}


def screen_seed(ready, q0, schedule, config, seed, output, fingerprint, reverse_order=False, progress=None):
    fit, validation, specs = ready['fit'], ready['screen_validation'], ready['specs']
    fit_docs = sorted(set(fit[0].metadata['document_ids']))
    val_docs = sorted(set(validation[0].metadata['document_ids']))
    if (set(fit_docs) & set(val_docs) or any(set(p.metadata['document_ids']) != set(fit_docs) for p in fit)
            or any(set(p.metadata['document_ids']) != set(val_docs) for p in validation)
            or {p.panel_id for p in fit} != {p.panel_id for p in validation}):
        raise ValueError('Each arm requires the same document split and sites')
    sources = {p.metadata['source_type'] for p in [*fit, *validation]}
    if len(sources) != 1 or not sources <= {'real_layer', 'unit_test_fixture'}:
        raise ValueError('Inconsistent screen panel provenance')
    source = sources.pop()
    fit_hash = panel_fingerprint(fit)
    schedule_hash = config_hash(schedule)
    rows, refs = [], []
    arms = tuple(reversed(ARMS)) if reverse_order else ARMS
    for objective, method in ((OLD, HAD), *arms):
        dest = Path(output) / f'seed_{seed}' / objective / method
        cfg = replace(pilot_solver_config(config, execution_fingerprint=fingerprint), fit_objective=objective)
        def emit(phase, **details):
            report(progress, 'objective.' + phase, seed=seed, objective=objective, method=method, **details)
        emit('start', pair_count=len(schedule))
        solver = fit_baseline(method, fit, q0, specs, schedule, cfg, progress=emit)
        q = solver.pop('q')
        row = {'source_type': source, 'evidence_kind': 'FIT_OBJECTIVE_SCREEN', 'seed': seed,
               'objective': objective, 'method': method, 'fit_documents': fit_docs,
               'validation_documents': val_docs, 'fit_panel_sha256': fit_hash, 'schedule_sha256': schedule_hash,
               'parent_manifest_sha256': ready['parent_manifest_sha256'], 'solver': solver,
               'validation': panel_metrics(validation, q, specs, cfg.device) if solver['status']=='COMPLETE' else []}
        path = atomic_json(dest / 'objective_result.json', row)
        refs.append({'path': str(path.resolve()), 'sha256': sha256_file(path)})
        del q
        if solver['status'] != 'COMPLETE':
            return rows, refs, solver.get('reason') or solver['status']
        rows.append(row)
    return rows, refs, None


def run_objective_screen(config, prepared, output, *, execution_fingerprint, progress=None):
    validate_screen_config(config)
    all_rows, refs = [], []
    device, size = config['solver']['device'], prepared['hidden_size']
    for idx, seed in enumerate(config['seeds']):
        q0 = hadamard_rotation(size, seed, device)
        prefix = pair_schedule(size, config['solver']['pair_budget'], seed)[:SCREEN_PAIRS]
        ready = prepared['seeds'].pop(seed)
        rows, paths, reason = screen_seed(ready, q0, prefix, config, seed, Path(output)/'objective_screen',
                                         execution_fingerprint, reverse_order=bool(idx), progress=progress)
        refs.extend(paths)
        if reason:
            return {'status': 'RESOURCE_BLOCKED', 'reason': reason, 'screen_artifacts': refs,
                    'objective_screen': {'decision': 'INSUFFICIENT_EVIDENCE'}}
        all_rows.extend(rows)
        del ready, q0
    source_types = {r['source_type'] for r in all_rows}
    if len(source_types) != 1:
        raise ValueError('Inconsistent source types across seeds')
    source = source_types.pop()
    assessment = assess_screen(all_rows)
    if source == 'unit_test_fixture':
        assessment['fixture_numerical_decision'] = assessment['decision']
        assessment['decision'] = 'FIXTURE_ONLY'
    result = {'status': 'SCREEN_COMPLETE', 'evidence_kind': 'FIT_OBJECTIVE_SCREEN',
              'source_type': source, 'screen_artifacts': refs, 'objective_screen': assessment}
    atomic_json(Path(output)/'objective_screen_summary.json', result)
    return result


def verified_screen(result_dir, parent_config):
    """Verify a completed real screen before freezing the one formal confirmation."""
    root = Path(result_dir).resolve()
    result = read_json(root/'result.json')
    if (result.get('status') != 'SCREEN_COMPLETE' or result.get('source_type') != 'real_layer'
            or result.get('evidence_kind') != 'FIT_OBJECTIVE_SCREEN'
            or result['config_sha256'] != config_hash(parent_config)):
        raise ValueError('Expected a completed real screen on this frozen parent config')
    verify_parent_manifests(parent_config)
    rows = []
    expected_docs = {}
    for seed in parent_config['seeds']:
        path = parent_config['data']['token_manifests'][str(seed)]
        _, parts = load_token_manifest(path)
        ids = [b['document_id'] for b in parts['fit']]
        expected_docs[seed] = (sorted(ids[:4]), sorted(ids[4:]), sha256_file(path))
    for ref in result['screen_artifacts']:
        # Allow a restored run directory; bind the relative seed/objective/method path and bytes.
        tail = Path(ref['path'].replace('\\','/')).parts[-4:]
        if len(tail) != 4 or tail[3] != 'objective_result.json' or tail[1] not in (OLD, NEW) or tail[2] not in (CELL,GRID,HAD):
            raise ValueError('Unexpected screen artifact path')
        if tail[0] not in {f'seed_{s}' for s in parent_config['seeds']}:
            raise ValueError('Unexpected screen artifact seed')
        path = root/'objective_screen'/Path(*tail)
        if sha256_file(path) != ref['sha256']:
            raise ValueError('Screen artifact checksum mismatch')
        row = read_json(path)
        if (row['source_type'] != 'real_layer' or row['solver']['status'] != 'COMPLETE'
                or row['solver']['requested_pairs'] != SCREEN_PAIRS or row['evidence_kind'] != 'FIT_OBJECTIVE_SCREEN'
                or (row['method'] != HAD and row['solver']['completed_pairs'] != SCREEN_PAIRS)
                or (f"seed_{row['seed']}", row['objective'], row['method']) != tail[:3]
                or row['solver']['fit_objective']['mode'] != row['objective']
                or len(row['validation']) != 9
                or (row['fit_documents'],row['validation_documents'],row['parent_manifest_sha256']) != expected_docs[row['seed']]):
            raise ValueError('Screen artifacts and original fit manifest do not match')
        rows.append(row)
    check = assess_screen(rows)
    for seed in parent_config['seeds']:
        same_seed = [r for r in rows if r['seed']==seed]
        for key in ('fit_panel_sha256','schedule_sha256'):
            if len({r[key] for r in same_seed})!=1:
                raise ValueError('Unmatched screen inputs or coordinate schedule')
        if len({config_hash(r['solver']['limits']) for r in same_seed})!=1:
            raise ValueError('Unmatched screen risk budgets')
        for mode in (OLD,NEW):
            if len({config_hash(r['solver']['fit_objective']) for r in same_seed if r['objective']==mode})!=1:
                raise ValueError('Unmatched screen fit objectives')
    if (len(rows) != 5 * len(parent_config['seeds']) or check != result['objective_screen']
            or check['decision'] != 'TEST_ONE_G1'):
        raise ValueError('Objective screen does not support the formal confirmation')
    return sha256_file(root/'result.json')
