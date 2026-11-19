"""Measure a frozen batch fit on new documents, with the parent grid and fixed random control."""
import math
import shutil
from pathlib import Path

import numpy as np
import torch

from .data import confirmation_partitions
from .matching_batches import (BATCH_SIZE, BATCH_STUDY_KIND, load_matching_arms, q_digest,
                               random_batch_indices, replay_continued_grid)
from .matching_study import CONTINUED, HYBRID, paired_site_gains
from .output_matching import apply_matching
from .pilot import panel_metrics
from .progress import report
from .runtime import atomic_json, read_json, sha256_file
from .solver import panel_fingerprint


KIND = 'FROZEN_BATCH_DOCUMENT_PILOT'
SELECTED, RANDOM = 'accepted_batches', 'same_count_random'
DESIGN = {'version': 1, 'partition_namespace': 2026111902, 'documents_per_seed': 16,
          'tokens_per_document': 32, 'bootstrap_draws': 10000, 'bootstrap_seed': 20261119,
          'primary': 'accepted batches vs continued grid, median across all nine sites',
          'advance': 'both seed medians positive and paired document 95% lower bound positive'}


def locate_fit_screen(experiments, parent_hash, archive_hash):
    candidates = []
    for trace in Path(experiments).glob('*/seed_11/batch_fit.json'):
        folder = trace.parents[1]
        receipt = folder/'result.json'
        if receipt.is_file():
            result = read_json(receipt)
            if (result.get('evidence_kind') == BATCH_STUDY_KIND and result.get('config_sha256') == parent_hash
                    and result.get('archive_sha256') == archive_hash):
                candidates.append((result['created_at_utc'], folder))
    if not candidates:
        raise FileNotFoundError('Restore the completed fit screen (result.json and both seed_*/batch_fit.json) '
                                'or specify its directory with --screen-run.')
    return max(candidates)[1]


def verified_screen(folder, seeds, archive_hash, parent_hash):
    result = read_json(folder/'result.json')
    if (result.get('status') != 'DIAGNOSTIC_COMPLETE' or result.get('source_type') != 'real_layer'
            or result.get('evidence_kind') != BATCH_STUDY_KIND or result.get('next_step') != 'fresh_document_pilot'
            or result.get('archive_sha256') != archive_hash or result.get('config_sha256') != parent_hash
            or result.get('batch_size') != BATCH_SIZE or set(result['fit_screen']) != {str(s) for s in seeds}):
        raise ValueError('Fit screen did not qualify with these frozen G1 inputs and 01d archive')
    traces = {}
    for seed in seeds:
        summary = result['fit_screen'][str(seed)]
        path = folder/f'seed_{seed}'/'batch_fit.json'
        if sha256_file(path) != summary['fit_trace_sha256']:
            raise ValueError(f'Fit trace checksum mismatch for seed {seed}')
        row = read_json(path)
        loss_delta = row['final']['loss'] - row['best_whole_layer']['loss']
        output_delta = row['final']['output_loss'] - row['initial']['output_loss']
        selected = [step['batch'] for step in row['trace'] if step['accepted']]
        if (row['seed'] != seed or row['source_type'] != 'real_layer' or row['evidence_kind'] != BATCH_STUDY_KIND
                or row['archive_sha256'] != archive_hash or row['accepted_batches'] != selected
                or len(selected) != summary['accepted_batches'] or row['applied_pairs'] != summary['applied_pairs']
                or loss_delta != summary['delta_loss_vs_whole'] or output_delta != summary['delta_output_loss_vs_grid']
                or not (loss_delta < 0 and output_delta < 0)):
            raise ValueError(f'Fit trace and promotion decision disagree for seed {seed}')
        traces[seed] = row
    return result, traces


def freeze_batch_inputs(output, screen_dir, archive, seeds, parent_hash):
    """Copy the exact receipts so a later evaluation does not consult a changing run directory."""
    dest = output/'fit_reference'
    dest.mkdir()
    shutil.copyfile(archive, dest/'measurements.zip')
    shutil.copyfile(screen_dir/'result.json', dest/'result.json')
    for seed in seeds:
        sub = dest/f'seed_{seed}'
        sub.mkdir()
        shutil.copyfile(screen_dir/sub.name/'batch_fit.json', sub/'batch_fit.json')
    archive_hash = sha256_file(dest/'measurements.zip')
    result, _ = verified_screen(dest, seeds, archive_hash, parent_hash)
    return {'design': dict(DESIGN), 'parent_config_sha256': parent_hash, 'archive_sha256': archive_hash,
            'screen_run_id': result['run_id'], 'screen_result_sha256': sha256_file(dest/'result.json')}


def draw_fresh_documents(blocks, previous, reserved, arms_by_seed):
    prior_keys = set()
    for seed, arms in arms_by_seed.items():
        row = arms[CONTINUED]
        if set(row['fit_documents']) != {b['document_id'] for b in previous[seed]['fit']}:
            raise ValueError(f'01d fit documents do not match the parent for seed {seed}')
        prior_keys.update((b['document_id'], b['block_id']) for site in row['heldout'] for b in site['blocks'])
    old_blocks = [b for b in blocks if (b['document_id'], b['block_id']) in prior_keys]
    if len(old_blocks) != len(prior_keys) or not prior_keys:
        raise ValueError('Could not identify the actual 01d validation blocks in the pinned corpus')
    fresh = confirmation_partitions(blocks, previous, DESIGN['documents_per_seed'],
                                    namespace=DESIGN['partition_namespace'], exclude=[*reserved, *old_blocks])
    groups = {'original_fit': [b for p in previous.values() for b in p['fit']],
              'original_heldout': [b for p in previous.values() for b in p['dev']],
              'reserved_confirmation': reserved, 'observed_01d': old_blocks}
    keys = ('document_id', 'block_id', 'document_sha256', 'token_ids_sha256')
    return fresh, {name: [{k: b[k] for k in keys} for b in rows] for name, rows in groups.items()}


def load_frozen_fit(config):
    source = config['batch_document_pilot']
    folder = Path(config['protocol_manifest_path']).parent/'fit_reference'
    archive = folder/'measurements.zip'
    if sha256_file(archive) != source['archive_sha256'] or sha256_file(folder/'result.json') != source['screen_result_sha256']:
        raise ValueError('Frozen fit inputs changed after document selection')
    _, traces = verified_screen(folder, config['seeds'], source['archive_sha256'], source['parent_config_sha256'])
    return archive, traces


def verify_document_protocol(config):
    from .objective_screen import verify_parent_manifests
    verify_parent_manifests(config)
    protocol = Path(config['protocol_manifest_path'])
    frozen = read_json(protocol)
    source = config.get('batch_document_pilot', {})
    fixed = ('batch_document_pilot', 'formal_pilot', 'model_revision', 'tokenizer_revision', 'dataset_revision',
             'quantization', 'rotation', 'solver', 'runtime')
    if (source.get('design') != DESIGN or config['seeds'] != [11, 23] or frozen.get('status') != 'FROZEN'
            or frozen.get('model_id') != config.get('model_id', config['formal_pilot']['model_id'])
            or any(key not in frozen or frozen[key] != config.get(key) for key in fixed)):
        raise ValueError('Fresh document protocol changed after preparation')
    load_frozen_fit(config)
    prior_file = protocol.parent/'prior_documents.json'
    if sha256_file(prior_file) != source['prior_documents_sha256']:
        raise ValueError('Prior document exclusions changed')
    prior = [b for group in read_json(prior_file).values() for b in group]
    keys = ('document_id', 'document_sha256', 'token_ids_sha256')
    seen = {k: {b[k] for b in prior} for k in keys}
    for seed in config['seeds']:
        manifest = read_json(config['data']['token_manifests'][str(seed)])
        rows = manifest['partitions']['dev']
        identity = manifest['identity']
        if (identity['source_split'] != 'train' or identity['seed'] != seed or len(rows) != DESIGN['documents_per_seed']
                or any(identity[k] != config[k] for k in ('model_revision', 'tokenizer_revision', 'dataset_revision'))):
            raise ValueError('Unexpected corpus split or fresh document count')
        for k in keys:
            values = {b[k] for b in rows}
            if values & seen[k] or len(values) != len(rows):
                raise ValueError(f'Fresh cohort reuses {k}')
            seen[k].update(values)


def frozen_rotations(arms, trace, dimension, seed, device):
    grid = replay_continued_grid(arms, dimension, seed, device)
    parent = arms[CONTINUED]
    if (trace['parent_q_sha256'] != parent['q_sha256'] or trace['fit_panel_sha256'] != parent['fit_panel_sha256']
            or trace['fit_documents'] != parent['fit_documents']):
        raise ValueError(f'Fit trace parent differs from 01d for seed {seed}')
    proposal = arms[HYBRID]['solver']
    pairs = np.asarray(proposal['pairs'])
    angles = grid.new_tensor(proposal['proposed_angles'])
    blocks = [slice(i, i+BATCH_SIZE) for i in range(0, len(pairs), BATCH_SIZE)]
    kept, random = trace['accepted_batches'], trace['same_count_random']
    if (trace['batch_count'] != len(blocks) or trace['limits'] != proposal['limits']
            or kept != sorted(set(kept)) or any(type(i) != int or not 0 <= i < len(blocks) for i in kept)
            or random['batches'] != random_batch_indices(len(blocks), len(kept), seed)):
        raise ValueError(f'Invalid frozen batch indices for seed {seed}')
    selected = grid
    for i in kept:
        block = blocks[i]
        selected = apply_matching(selected, pairs[block], angles[block])
    if q_digest(selected) != trace['final_q_sha256']:
        raise ValueError(f'Accepted batch rotation checksum mismatch for seed {seed}')
    # Replay the random control with the same masked operation used during fitting.
    masked = torch.zeros_like(angles)
    for i in random['batches']:
        masked[blocks[i]] = angles[blocks[i]]
    if int(torch.count_nonzero(masked)) != random['nonzero_angles']:
        raise ValueError(f'Random control angle count differs for seed {seed}')
    return {CONTINUED: grid, SELECTED: selected, RANDOM: apply_matching(grid, pairs, masked)}


def assess_documents(records, seeds, placements):
    contrasts = {'selected_vs_grid': (SELECTED, CONTINUED), 'selected_vs_random': (SELECTED, RANDOM),
                 'random_vs_grid': (RANDOM, CONTINUED)}
    stats = paired_site_gains(records, seeds, placements, contrasts, repetitions=DESIGN['bootstrap_draws'],
                              rng_seed=DESIGN['bootstrap_seed'])
    primary = stats['comparisons']['selected_vs_grid']
    if 'undefined' in primary:
        return stats, 'insufficient_evidence'
    all_sites = primary['all']
    advance = min(all_sites['seed_medians_pct'].values()) > 0 and all_sites['paired_95_pct'][0] > 0
    site, gain = min(primary['site_gain_pct'].items(), key=lambda item: item[1])
    stats['worst_site_vs_grid'] = {'panel_id': site, 'gain_pct': gain}
    return stats, 'formal_g1_retest' if advance else 'stop_batch_variant'


@torch.inference_mode()
def measure_frozen_batches(prepared, arms_by_seed, traces, output, *, device, progress=None):
    seeds = sorted(traces)
    records, artifacts, calibration = [], [], {}
    seen_docs = set()
    placements = None
    for seed in seeds:
        panels = prepared['seeds'].pop(seed)
        dev, arms, trace = panels['heldout'], arms_by_seed[seed], traces[seed]
        actual = {p.panel_id: p.placement for p in dev}
        doc_ids = set(dev[0].metadata['document_ids'])
        if (len(actual) != 9 or (placements is not None and placements != actual)
                or len(doc_ids) != DESIGN['documents_per_seed'] or doc_ids & seen_docs
                or any(p.metadata['source_type'] != 'real_layer' or set(p.metadata['document_ids']) != doc_ids
                       or p.metadata['tokens_per_block'] != [DESIGN['tokens_per_document']]*len(doc_ids) for p in dev)):
            raise ValueError('Fresh validation panels do not match the frozen site and document design')
        placements = actual
        seen_docs.update(doc_ids)
        rotations = frozen_rotations(arms, trace, prepared['hidden_size'], seed, device)
        grid_seconds = arms[CONTINUED]['total_calibration_seconds']
        proposal_seconds = arms[HYBRID]['solver']['seconds']
        calibration[str(seed)] = {'continued_grid_seconds': grid_seconds, 'whole_proposal_and_check_seconds': proposal_seconds,
                                  'batch_screen_with_controls_seconds': trace['fit_screen_seconds'],
                                  'accounted_candidate_seconds': grid_seconds + proposal_seconds + trace['fit_screen_seconds'],
                                  'random_fit_feasible': trace['same_count_random']['feasible']}
        panel_sha = panel_fingerprint(dev)
        for method, q in rotations.items():
            report(progress, 'matching.fresh_documents', seed=seed, method=method)
            values = panel_metrics(dev, q, panels['specs'], device=device, error_terms=True)
            orth = float(torch.linalg.norm(q.T@q - torch.eye(len(q), dtype=q.dtype, device=q.device))/math.sqrt(len(q)))
            row = {'source_type': 'real_layer', 'evidence_kind': KIND, 'seed': seed, 'method': method,
                   'heldout_panel_sha256': panel_sha, 'q_sha256': q_digest(q), 'orthogonality_residual': orth,
                   'heldout_documents': sorted(doc_ids), 'heldout': values}
            path = atomic_json(output/f'seed_{seed}'/method/'document_result.json', row)
            artifacts.append({'path': path.relative_to(output).as_posix(), 'sha256': sha256_file(path)})
            records.append({'seed': seed, 'method': method, 'panels': {p['panel_id']: p['blocks'] for p in values}})
        del rotations, panels, dev, q
    stats, next_step = assess_documents(records, seeds, placements)
    return {'status': 'STUDY_COMPLETE', 'source_type': 'real_layer', 'evidence_kind': KIND,
            'design': dict(DESIGN), 'next_step': next_step, 'document_summary': stats,
            'calibration_seconds_by_seed': calibration, 'document_artifacts': artifacts}


def run_document_pilot(config, prepared, output, *, progress=None):
    archive, traces = load_frozen_fit(config)
    arms = load_matching_arms(archive, config['seeds'])
    result = measure_frozen_batches(prepared, arms, traces, output, device=config['solver']['device'], progress=progress)
    return {**result, 'fit_source': config['batch_document_pilot'], 'protocol_sha256': config['protocol_sha256']}
