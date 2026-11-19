"""Test smaller joint updates on the original 01d fit panels, reusing its fitted angles."""
import hashlib
import json
import math
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from .matching_study import CONTINUED, HAD, HYBRID, KIND, WARM
from .output_matching import apply_matching, check_time, fit_totals, fits_limits
from .pilot import pilot_solver_config
from .progress import report
from .runtime import atomic_json, sha256_file
from .solver import panel_fingerprint
from .tensor_solver import hadamard_rotation, rotate_rows, synchronize


BATCH_SIZE = 64
BATCH_STUDY_KIND = 'FIT_MATCHING_BATCH_SCREEN'


def load_fit_reference(archive_path, seeds):
    """Read the fit fields of four hash-verified arms per seed from a 01d archive."""
    records = {seed: {} for seed in seeds}
    with ZipFile(archive_path) as archive:
        prefix = 'run/' if 'run/matching_summary.json' in archive.namelist() else ''
        receipt = json.loads(archive.read(prefix + 'matching_summary.json'))
        if (receipt['status'] != 'STUDY_COMPLETE' or receipt['source_type'] != 'real_layer'
                or receipt['evidence_kind'] != KIND or not receipt['design'].get('refine_continued_grid')):
            raise ValueError('Expected a complete real-layer 01d study with the continued-grid arm')
        for ref in receipt['matching_artifacts']:
            parts = ref['path'].replace('\\', '/').split('/')[-3:]
            if len(parts) != 3 or parts[1] not in (HAD, WARM, CONTINUED, HYBRID):
                continue
            payload = archive.read(prefix + '/'.join(parts))
            if hashlib.sha256(payload).hexdigest() != ref['sha256']:
                raise ValueError(f'Matching record checksum mismatch: {parts}')
            row = json.loads(payload)
            seed, method = row['seed'], row['method']
            if (parts != [f'seed_{seed}', method, 'matching_result.json'] or seed not in records
                    or method in records[seed] or row['source_type'] != 'real_layer' or row['evidence_kind'] != KIND):
                raise ValueError('Archive method identity differs from the requested reference')
            records[seed][method] = {k: row[k] for k in ('solver', 'q_sha256', 'fit_panel_sha256', 'fit_documents')}
    for arms in records.values():
        if set(arms) != {HAD, WARM, CONTINUED, HYBRID} or arms[HYBRID]['solver']['parent'] != CONTINUED:
            raise ValueError('Archive is missing the required fit reference or its continued-grid parent')
    return records


def q_digest(q):
    return hashlib.sha256(q.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def check_replay(measured, recorded):
    for key, value in measured.items():
        if not math.isclose(value, recorded[key], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f'Frozen 01d fit replay differs at {key}: {value} vs {recorded[key]}')


@torch.inference_mode()
def screen_batches(fit, specs, q, reference, cfg, seed, *, batch_size=BATCH_SIZE, progress=None):
    """One sweep in the archived order. Materialize all input and output sites before accepting."""
    if batch_size <= 0:
        raise ValueError('Batch size must be positive')
    pairs = np.asarray(reference['pairs'], dtype=np.int64)
    angles = q.new_tensor(reference['proposed_angles'])
    limits = reference['limits']
    synchronize(q.device)
    started = time.perf_counter()
    deadline = time.monotonic() + cfg.wall_seconds
    start_sites, end_sites = {}, {}
    initial = fit_totals(fit, q, specs, site_nmse=start_sites)
    check_replay(initial, reference['initial'])
    trials = []
    for trial in reference['joint_fit_trials']:
        scale = trial['scale']
        actual = fit_totals(fit, apply_matching(q, pairs, angles * scale), specs) if scale else initial
        check_replay(actual, trial)
        feasible = fits_limits(actual, limits, cfg.budget_tolerance)
        if feasible != trial['feasible']:
            raise ValueError('Whole-layer replay changed scale-budget feasibility')
        trials.append(dict(scale=scale, feasible=feasible, **actual))
    best_whole = min((t for t in trials if t['feasible']), key=lambda t: t['loss'])
    blocks = [slice(i, i+batch_size) for i in range(0, len(pairs), batch_size)]
    kept, trace = [], []
    current_q, current = q, initial
    for index, block in enumerate(blocks):
        check_time(deadline)
        report(progress, 'matching.fit_batch', seed=seed, batch=index+1, batches=len(blocks))
        proposal = apply_matching(current_q, pairs[block], angles[block])
        actual = fit_totals(fit, proposal, specs)
        feasible = fits_limits(actual, limits, cfg.budget_tolerance)
        accepted = feasible and actual['loss'] < current['loss']
        trace.append({'batch': index, 'nonzero_angles': int(torch.count_nonzero(angles[block])),
                      'parent_loss': current['loss'], 'parent_output_loss': current['output_loss'],
                      'feasible': feasible, 'accepted': accepted, **actual})
        if accepted:
            kept.append(index)
            current_q, current = proposal, actual
    final = fit_totals(fit, current_q, specs, site_nmse=end_sites)
    check_replay(final, current)
    # Use the same number of full batches, selected independently of their losses.
    rng = np.random.default_rng(np.random.SeedSequence([2026111901, seed]))
    random_batches = sorted(rng.choice(len(blocks), len(kept), replace=False).tolist())
    mask_angles = torch.zeros_like(angles)
    for index in random_batches:
        block = blocks[index]
        mask_angles[block] = angles[block]
    random = fit_totals(fit, apply_matching(q, pairs, mask_angles), specs)
    synchronize(q.device)
    check_time(deadline)
    return {'initial': initial, 'whole_layer_trials': trials, 'best_whole_layer': best_whole,
            'limits': limits, 'trace': trace, 'accepted_batches': kept, 'batch_count': len(blocks),
            'applied_pairs': sum(trace[i]['nonzero_angles'] for i in kept), 'final': final,
            'initial_site_nmse': start_sites, 'final_site_nmse': end_sites, 'final_q_sha256': q_digest(current_q),
            'same_count_random': {'batches': random_batches, 'nonzero_angles': int(torch.count_nonzero(mask_angles)),
                                  'feasible': fits_limits(random, limits, cfg.budget_tolerance), **random},
            'fit_screen_seconds': time.perf_counter()-started}


def run_batch_screen(config, prepared, output, archive_path, *, progress=None):
    seeds = config['seeds']
    archive_sha = sha256_file(archive_path)
    references = load_fit_reference(archive_path, seeds)
    cfg = pilot_solver_config(config)
    summary = {}
    for seed in seeds:
        panels = prepared['seeds'].pop(seed)
        arms = references[seed]
        fingerprint = panel_fingerprint(panels['fit'])
        documents = sorted(set(panels['fit'][0].metadata['document_ids']))
        if (any(p.metadata.get('source_type') != 'real_layer' for p in panels['fit']) or
                any(r['fit_panel_sha256'] != fingerprint or r['fit_documents'] != documents for r in arms.values())):
            raise ValueError(f'Recaptured seed {seed} fit panels differ from the archived 01d inputs')
        q = hadamard_rotation(prepared['hidden_size'], seed, cfg.device)
        for method in (HAD, WARM, CONTINUED):
            for factor in arms[method]['solver']['factors']:
                rotate_rows(q, factor['pair'], factor['theta'])
            if q_digest(q) != arms[method]['q_sha256']:
                raise ValueError(f'Seed {seed} failed to reproduce the archived {method} rotation')
        record = screen_batches(panels['fit'], panels['specs'], q, arms[HYBRID]['solver'], cfg, seed, progress=progress)
        path = atomic_json(Path(output)/f'seed_{seed}'/'batch_fit.json', {
            'source_type': 'real_layer', 'evidence_kind': BATCH_STUDY_KIND, 'seed': seed,
            'fit_documents': documents, 'fit_panel_sha256': fingerprint,
            'parent_q_sha256': arms[CONTINUED]['q_sha256'], 'archive_sha256': archive_sha, **record})
        final, initial, whole = record['final'], record['initial'], record['best_whole_layer']
        summary[str(seed)] = {'accepted_batches': len(record['accepted_batches']), 'applied_pairs': record['applied_pairs'],
                              'delta_loss_vs_whole': final['loss']-whole['loss'],
                              'delta_output_loss_vs_grid': final['output_loss']-initial['output_loss'],
                              'fit_trace_sha256': sha256_file(path)}
    advance = all(s['delta_loss_vs_whole'] < 0 and s['delta_output_loss_vs_grid'] < 0 for s in summary.values())
    return {'status': 'DIAGNOSTIC_COMPLETE', 'source_type': 'real_layer', 'evidence_kind': BATCH_STUDY_KIND,
            'batch_size': BATCH_SIZE, 'archive_sha256': archive_sha, 'fit_screen': summary,
            'next_step': 'fresh_document_pilot' if advance else 'stop_batch_variant'}
