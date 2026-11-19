"""Freeze the qualifying A100 rotations and draw a paired WikiText training-article holdout."""
import hashlib
import json
import operator
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .bilateral_screen import DESIGN as FIT_DESIGN, DOCUMENT_DESIGN, retest_grid_rows
from .data import confirmation_partitions
from .gates import load_rules
from .matching_study import HAD
from .objective_screen import verify_parent_manifests
from .runtime import config_hash, read_json, sha256_file


DESIGN = {**DOCUMENT_DESIGN, 'partition_namespace': 2026111905, 'bootstrap_seed': 2026111905,
          'watch_site': 'layer_0.up_proj',
          'decision': 'positive document interval and both seed medians vs output-only; at least one G1 route still possible against grid on measured quality'}
IDENTITY = ('block_id', 'document_id', 'document_sha256', 'token_ids_sha256')
FIT_FILE = 'bilateral_fit_7b.zip'
GRID_FILE = 'batch_g1_retest_7b.zip'
COHORT_FILE = 'bilateral_document_cohort_7b.json'


def block_identity(row):
    return {key: row[key] for key in IDENTITY}


def checked_json(archive, name, digest):
    data = archive.read(name)
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'Frozen archive checksum mismatch: {name}')
    return json.loads(data)


def read_7b_fit(screen, grid):
    """Bind the Colab run/result layout to its raw two-seed traces, grid and token identities."""
    with ZipFile(screen) as archive:
        receipt = json.loads(archive.read('run/result.json'))
        config = json.loads(archive.read('run/config.json'))
        protocol = checked_json(archive, 'inputs/protocol.json', config['protocol_sha256'])
        retest = config['batch_g1_retest']
        if (receipt['status'] != 'DIAGNOSTIC_COMPLETE' or receipt['execution_mode'] != 'bilateral_angle_fit_screen'
                or receipt['source_type'] != 'real_layer' or receipt['evidence_kind'] != 'FIT_BILATERAL_ANGLE_SCREEN'
                or receipt['design'] != FIT_DESIGN or receipt['next_step'] != 'fresh_document_pilot'
                or config_hash(config) != receipt['config_sha256'] or config['seeds'] != FIT_DESIGN['seeds']
                or receipt['reference_archive_sha256'] != sha256_file(grid) or protocol['batch_g1_retest'] != retest):
            raise ValueError('Expected a promoted A100 fit screen tied to the completed grid retest')
        grids = retest_grid_rows(grid, config, manifest_hashes=protocol['token_manifest_sha256'])
        old = checked_json(archive, 'inputs/prior_documents.json', retest['excluded_sha256'])
        traces, parts, fingerprints = {}, {}, set()
        for seed, ref in grids.items():
            summary = receipt['by_seed'][str(seed)]
            trace = checked_json(archive, f'run/seed_{seed}/angle_screen.json', summary['trace_sha256'])
            manifest = checked_json(archive, f'inputs/seed_{seed}/data_manifest.json', ref['data_manifest_sha256'])
            fingerprints.add(manifest['identity']['dataset_fingerprint'])
            parts[seed] = manifest['partitions']
            output_only, bilateral = (trace['arms'][k]['selection'] for k in FIT_DESIGN['arms'])
            full = trace['full_grid_site_nmse']
            gains = {site: 100 * (1 - bilateral['final_site_nmse'][site] / value) for site, value in full.items()}
            within = sum(v >= -2 for v in gains.values())
            if (trace['seed'] != seed or trace['prefix_grid_pairs'] != 64 or trace['fit_panel_sha256'] != ref['fit_panel_fingerprint']
                    or trace['fit_documents'] != ref['fit_documents'] or set(ref['fit_documents']) != {r['document_id'] for r in parts[seed]['fit']}
                    or trace['summary'] != {k:v for k,v in summary.items() if k != 'trace_sha256'}
                    or set(trace['arms']) != set(FIT_DESIGN['arms']) or len(gains) != 9
                    or gains != summary['gain_vs_full_grid_by_site_pct'] or within != summary['sites_within_two_pct_of_full_grid']
                    or float(np.median(list(gains.values()))) != summary['median_fit_gain_vs_full_grid_pct']
                    or bilateral['final'] != summary['bilateral_fit'] or output_only['final'] != summary['output_only_fit']
                    or not bilateral['final']['loss'] < output_only['final']['loss'] or within < 6):
                raise ValueError(f'The fit promotion is inconsistent for seed {seed}')
            traces[seed] = trace
    if len(fingerprints) != 1:
        raise ValueError('Fitted seeds used different corpus fingerprints')
    return {'config': config, 'receipt': receipt, 'grids': grids, 'traces': traces, 'parts': parts,
            'prior': old, 'dataset_fingerprint': fingerprints.pop()}


def draw_documents(blocks, fit, small_model_blocks, *, design=DESIGN):
    """Use the same fitted blocks as the A100 screen, excluding all previous evaluation documents."""
    by_id = {b['block_id']: b for b in blocks}
    previous = {}
    for seed, parts in fit['parts'].items():
        previous[seed] = {}
        for split in ('fit', 'dev'):
            originals = []
            for row in parts[split]:
                actual = by_id.get(row['block_id'])
                if actual is None or block_identity(actual) != block_identity(row):
                    raise ValueError('Frozen A100 block differs from the pinned corpus or tokenizer')
                originals.append(actual)
            previous[seed][split] = originals
    return confirmation_partitions(blocks, previous, design['documents_per_seed'], namespace=design['partition_namespace'],
                                   exclude=[*fit['prior'], *small_model_blocks])


def verify_document_inputs(config, *, row_crossfit=False):
    verify_parent_manifests(config)
    directory = Path(config['protocol_manifest_path']).parent
    protocol = read_json(directory/'protocol.json')
    frozen = config['bilateral_document_pilot']
    fixed = ('bilateral_document_pilot','formal_pilot','model_revision','tokenizer_revision','dataset_revision',
             'solver','rotation','quantization','runtime')
    if (protocol['status'] != 'FROZEN' or any(protocol[k] != config[k] for k in fixed)
            or frozen['design'] != DESIGN or config['seeds'] != FIT_DESIGN['seeds']
            or set(frozen['archives']) != {FIT_FILE, GRID_FILE, COHORT_FILE}
            or frozen['quality_rules_sha256'] != config_hash(load_rules())):
        raise ValueError('Bilateral document protocol changed after preparation')
    folder = directory/'fit_reference'
    for filename, digest in frozen['archives'].items():
        if sha256_file(folder/filename) != digest:
            raise ValueError(f'Bilateral document input changed: {filename}')
    fit = read_7b_fit(folder/FIT_FILE, folder/GRID_FILE)
    cohort = read_json(folder/COHORT_FILE)
    if (cohort['fit_archive_sha256'] != frozen['archives'][FIT_FILE] or cohort['design'] != DESIGN
            or frozen['parent_config_sha256'] != fit['config']['batch_g1_retest']['parent_config_sha256']
            or any(config[k] != fit['config'][k] for k in ('formal_pilot','solver','rotation','quantization',
                                                           'model_revision','tokenizer_revision','dataset_revision'))
            or config['runtime']['dtype'] != fit['config']['runtime']['dtype']):
        raise ValueError('Frozen evaluation differs from the fitted model or panel protocol')
    old = [*fit['prior'], *cohort['previous_42m_train_blocks'],
           *(b for p in fit['parts'].values() for rows in p.values() for b in rows)]
    heldout = cohort['heldout']
    if bool(config.get('row_crossfit')) != row_crossfit:
        raise ValueError('Use the runner bound to this document protocol')
    if row_crossfit:
        from .row_crossfit import COHORT_FILE as crossfit_file, DESIGN as crossfit_design
        spec = config['row_crossfit']
        fresh = read_json(folder/crossfit_file)
        if (spec != protocol.get('row_crossfit') or spec['design'] != crossfit_design or fresh['design'] != crossfit_design
                or sha256_file(folder/crossfit_file) != spec['cohort_sha256'] or fresh['previous_cohort_hash'] != config_hash(cohort)):
            raise ValueError('Changed row-crossfit cohort or design')
        old.extend(b for rows in heldout.values() for b in rows)
        heldout = fresh['heldout']
    seen = {k:{b[k] for b in old} for k in IDENTITY}
    for seed in config['seeds']:
        actual = read_json(config['data']['token_manifests'][str(seed)])
        ident, parts = actual['identity'], actual['partitions']
        if (ident['seed'] != seed or ident['source_split'] != 'train'
                or ident['dataset_fingerprint'] != cohort['dataset_fingerprint']
                or any(ident[k] != config[k] for k in ('model_revision','tokenizer_revision','dataset_revision'))
                or [block_identity(b) for b in parts['fit']] != [block_identity(b) for b in fit['parts'][seed]['fit']]
                or [block_identity(b) for b in parts['dev']] != heldout[str(seed)]):
            raise ValueError(f'Document identity differs for seed {seed}')
        for key in IDENTITY:
            values = {b[key] for b in parts['dev']}
            if len(values) != DESIGN['documents_per_seed'] or values & seen[key]:
                raise ValueError(f'Held-out articles reused {key}')
            seen[key].update(values)
    return fit


def document_decision(stats, rules=None):
    """Apply only observable quality bounds, without imputing a matched baseline or runtime."""
    comparisons = stats['comparisons']
    if any('undefined' in v for v in comparisons.values()):
        return {'next_step': 'insufficient_evidence'}
    grid, primary = comparisons['vs_full_grid'], comparisons['vs_output_only']['all']
    site_gains = grid['site_gain_pct']
    gains = np.asarray(list(site_gains.values()))
    worst = min(site_gains, key=site_gains.get)
    mechanism = min(primary['seed_medians_pct'].values()) > 0 and primary['paired_95_pct'][0] > 0
    new = FIT_DESIGN['arms'][1]
    metrics = {'each_seed_pooled_heldout_improved': all(r[new] < r[HAD] for r in stats['mean_site_nmse_by_seed'].values()),
               'median_nmse_gain_vs_hadamard_pct': comparisons['vs_hadamard']['all']['median_gain_pct'],
               'max_site_nmse_regression_vs_best_pct': float(-gains.min()),
               'median_nmse_gain_vs_best_matched_solver_pct': grid['all']['median_gain_pct'],
               'median_nmse_regression_vs_best_matched_solver_pct': -grid['all']['median_gain_pct'],
               'site_win_fraction_vs_best_matched_solver': float(np.mean(gains > 0)),
               'site_noninferior_fraction_at_2pct': float(np.mean(gains >= -2))}
    g1 = next(row for row in (rules or load_rules())['gates'] if row['gate_id'] == 'G1')
    ops = {'le':operator.le, 'ge':operator.ge, 'eq':operator.eq}
    violations = {route: [check['metric'] for check in [*g1['common_checks'], *checks]
                          if check['metric'] in metrics and not ops[check['op']](metrics[check['metric']], check['value'])]
                  for route, checks in g1['routes'].items()}
    next_step = 'review_inconclusive_pilot'
    if all(violations.values()):
        next_step = 'stop_bilateral_variant'
    elif mechanism:
        next_step = 'formal_g1_retest'
    return {'mechanism_replicated': mechanism, 'worst_site_vs_grid': {'site': worst, 'gain_pct': site_gains[worst]},
            'watched_site': DESIGN['watch_site'], 'watched_site_vs_grid': grid[DESIGN['watch_site']],
            'g1_quality_bounds_using_grid': metrics, 'failed_quality_bounds_by_route': violations, 'next_step':next_step}
