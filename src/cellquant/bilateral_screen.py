"""Compare input-aware and output-only angle proposals on the original calibration articles."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from .batch_retest import fit_candidate
from .matching_batches import BATCH_SIZE, check_replay, load_matching_arms, q_digest, replay_continued_grid, replay_selected_batches
from .matching_study import CONTINUED, HAD, WARM, paired_site_gains
from .output_matching import disjoint_pairs, fit_totals
from .pilot import panel_metrics, pilot_solver_config
from .progress import report
from .runtime import atomic_json, config_hash, read_json, sha256_file
from .solver import panel_fingerprint
from .tensor_solver import hadamard_rotation, rotate_rows


DESIGN = {'warm_grid_fraction': 0.5, 'matching_grid_points': 17, 'batch_size': BATCH_SIZE,
          'step_scales': [0., .25, .5, 1.], 'seeds': [11, 23],
          'arms': ['product', 'bilateral_fixed_scale_proposal'],
          'promote': 'both seeds beat the same-prefix output-only joint fit loss and keep at least six of nine fit sites within 2% of the full grid'}
DOCUMENT_DESIGN = {'version': 1, 'partition_namespace': 2026111904, 'documents_per_seed': 16, 'tokens_per_document': 32,
                   'bootstrap_draws': 10000, 'bootstrap_seed': 2026111904,
                   'primary': 'nine-site median vs output-only at the same prefix', 'quality_reference': CONTINUED}


def prefix_rotation(dimension, seed, warm, device):
    prefix_count = int(len(warm['log']) * DESIGN['warm_grid_fraction'])
    if (prefix_count <= 0 or warm['status'] != 'COMPLETE' or len(warm['log']) != warm['requested_pairs']
            or any(row['pair_index'] != i for i, row in enumerate(warm['log']))):
        raise ValueError('Invalid grid prefix in 01d archive')
    prefix = hadamard_rotation(dimension, seed, device)
    for row in warm['log'][:prefix_count]:
        if row['accepted']:
            rotate_rows(prefix, row['pair'], row['best']['theta'])
    return prefix, prefix_count


def retest_grid_rows(archive_path, config, *, manifest_hashes=None):
    """Read the two real parent grids from the bound-stopped 7B calibration cohort."""
    with ZipFile(archive_path) as archive:
        result = json.loads(archive.read('run/result.json'))
        previous = json.loads(archive.read('run/config.json'))
        if (result['execution_mode'] != 'batch_g1_retest' or result['source_type'] != 'real_layer'
                or result['status'] != 'RETEST_STOPPED_BY_BOUND' or not result['route_bounds']['both_routes_ruled_out']
                or config_hash(previous) != result['config_sha256']
                or previous['batch_g1_retest'] != config['batch_g1_retest']
                or previous['formal_pilot'] != config['formal_pilot'] or previous['quantization'] != config['quantization']):
            raise ValueError('Grid reference differs from the stopped 7B retest')
        rows = {}
        for seed in config['seeds']:
            row = json.loads(archive.read(f'run/seed_{seed}/{CONTINUED}/pilot_result.json'))
            manifest_sha = (manifest_hashes[str(seed)] if manifest_hashes is not None else
                            sha256_file(config['data']['token_manifests'][str(seed)]))
            if (row['seed'] != seed or row['method'] != CONTINUED or row['status'] != 'COMPLETE'
                    or row['evidence_kind'] != 'FORMAL_BATCH_G1' or row['model_revision'] != config['model_revision']
                    or manifest_sha != row['data_manifest_sha256']):
                raise ValueError('Retest grid seed, model or fitted articles differ')
            rows[seed] = row
        return rows


def replay_retest_parent(row, device):
    grid = hadamard_rotation(row['hidden_size'], row['seed'], device)
    for stage in row['solver']['grid_stages']:
        if stage['status'] != 'COMPLETE' or stage['completed_pairs'] != 128:
            raise ValueError('The 7B reference requires two completed 128-pair grids')
        for factor in stage['factors']:
            rotate_rows(grid, factor['pair'], factor['theta'])
    if q_digest(grid) != row['parent_q_sha256']:
        raise ValueError('Replayed 256-pair grid differs from the stopped 7B retest')
    return grid


@torch.inference_mode()
def run_screen(config, prepared, output, archive, *, progress=None):
    seeds = config['seeds']
    if seeds != DESIGN['seeds'] or config['solver'].get('fit_objective', 'sum_nmse') != 'sum_nmse':
        raise ValueError('The input-aware diagnostic uses both original seeds and the sum-NMSE objective')
    cfg = pilot_solver_config(config)
    retest = retest_grid_rows(archive, config) if config.get('batch_g1_retest') else None
    arms = load_matching_arms(archive, seeds) if retest is None else None
    by_seed = {}
    for seed in seeds:
        ready = prepared['seeds'][seed]
        fit, specs = ready['fit'], ready['specs']
        dimension = prepared['hidden_size']
        if retest is None:
            ref = arms[seed][WARM]
            fit_sha = ref['fit_panel_sha256']
            warm = ref['solver']
            expected_final = arms[seed][CONTINUED]['solver']['final']
            grid = replay_continued_grid(arms[seed], dimension, seed, cfg.device)
        else:
            ref = retest[seed]
            fit_sha = ref['fit_panel_fingerprint']
            warm, tail = ref['solver']['grid_stages']
            expected_final = tail['final']
            grid = replay_retest_parent(ref, cfg.device)
        if ('heldout' in ready or len(fit) != 9 or panel_fingerprint(fit) != fit_sha
                or not all(p.metadata['source_type'] == 'real_layer'
                           and set(p.metadata['document_ids']) == set(ref['fit_documents']) for p in fit)):
            raise ValueError('Expected only the nine real fit panels of the reference grid')
        full_sites = {}
        full_fit = fit_totals(fit, grid, specs, site_nmse=full_sites)
        check_replay({k: full_fit[k] for k in ('loss','budget_w','budget_a')}, expected_final)
        del grid
        prefix, prefix_count = prefix_rotation(dimension, seed, warm, cfg.device)
        actual = fit_totals(fit, prefix, specs)
        check_replay({k: actual[k] for k in ('loss','budget_w','budget_a')}, warm['log'][prefix_count-1]['best'])
        record = {'seed': seed, 'prefix_grid_pairs': prefix_count, 'prefix_q_sha256': q_digest(prefix),
                  'fit_documents': ref['fit_documents'], 'fit_panel_sha256': fit_sha,
                  'initial_fit': actual, 'full_grid_fit': full_fit, 'full_grid_site_nmse': full_sites, 'arms': {}}
        pairs = disjoint_pairs(dimension, seed)
        for criterion in (DESIGN['arms'] if seed == seeds[0] else reversed(DESIGN['arms'])):
            report(progress, 'bilateral.proposal', seed=seed, criterion=criterion, prefix_pairs=prefix_count)
            _, _, seconds, proposal, trace = fit_candidate(fit, specs, prefix, pairs, warm['limits'], cfg, 0.,
                                                          seed, DESIGN, progress, criterion=criterion)
            record['arms'][criterion] = {'proposal': proposal, 'selection': trace, 'incremental_solver_seconds': seconds}
        old, new = (record['arms'][c]['selection'] for c in DESIGN['arms'])
        if any(n <= 0 for n in full_sites.values()):
            raise ValueError('Fit comparison undefined for zero full-grid NMSE')
        vs_grid = {site: 100 * (1 - new['final_site_nmse'][site]/value) for site, value in full_sites.items()}
        within = sum(g >= -2 for g in vs_grid.values())
        compact = {'prefix_grid_pairs': prefix_count, 'output_only_fit': old['final'], 'bilateral_fit': new['final'],
                   'full_grid_fit': full_fit, 'prefix_fit': actual, 'gain_vs_full_grid_by_site_pct': vs_grid,
                   'median_fit_gain_vs_full_grid_pct': float(np.median(list(vs_grid.values()))),
                   'sites_within_two_pct_of_full_grid': within,
                   'accepted_batches': {c: arm['selection']['accepted_batches'] for c, arm in record['arms'].items()},
                   'incremental_solver_seconds': {c: arm['incremental_solver_seconds'] for c, arm in record['arms'].items()},
                   'lower_joint_loss_than_output_only': new['final']['loss'] < old['final']['loss']}
        record['summary'] = compact
        path = atomic_json(Path(output)/f'seed_{seed}'/'angle_screen.json', record)
        by_seed[str(seed)] = {**compact, 'trace_sha256': sha256_file(path)}
    advance = all(r['lower_joint_loss_than_output_only'] and r['sites_within_two_pct_of_full_grid'] >= 6
                  for r in by_seed.values())
    return {'status': 'DIAGNOSTIC_COMPLETE', 'evidence_kind': 'FIT_BILATERAL_ANGLE_SCREEN', 'source_type': 'real_layer',
            'design': DESIGN, 'reference_archive_sha256': sha256_file(archive), 'by_seed': by_seed,
            'next_step': 'fresh_document_pilot' if advance else 'stop_bilateral_proposal',
            'scope': 'fit-only development diagnostic; incremental timing excludes the reconstructed grid prefix'}


def replay_angle_arms(prefix, selection):
    if q_digest(prefix) != selection['prefix_q_sha256'] or set(selection['arms']) != set(DESIGN['arms']):
        raise ValueError('Grid prefix or angle arms differ from the frozen fit')
    rotations = {}
    for criterion, arm in selection['arms'].items():
        proposal = arm['proposal']
        rotations[criterion] = replay_selected_batches(prefix, np.asarray(proposal['pairs']),
            prefix.new_tensor(proposal['proposed_angles']), BATCH_SIZE, arm['selection'])
    return rotations


@torch.inference_mode()
def score_frozen(config, prepared, output, grid_archive, fit_archive, *, fit_7b=None, progress=None,
                 document_reference=None, full_up_weight=None):
    seeds = DESIGN['seeds']
    original, new = DESIGN['arms']
    device, dimension = config['solver']['device'], prepared['hidden_size']
    design = config['bilateral_document_pilot']['design'] if fit_7b is not None else DOCUMENT_DESIGN
    if (document_reference is None) != (full_up_weight is None):
        raise ValueError('Row diagnosis requires both the original document archive and the full up projection')
    row_seeds = {}
    if document_reference is not None:
        from .bilateral_rows import PREFIX, SITE, TILE_ROWS, read_reference, score_seed, summarize_rows
        document_rows = read_reference(document_reference, grid_archive, fit_archive, design, seeds, (new,CONTINUED))
    if fit_7b is None:
        refs = load_matching_arms(grid_archive, seeds)
        grids = {s:refs[s][CONTINUED] for s in seeds}
        with ZipFile(fit_archive) as archive:
            receipt = json.loads(archive.read('result.json'))
            if (receipt['reference_archive_sha256'] != sha256_file(grid_archive) or receipt['design'] != DESIGN
                    or receipt['next_step'] != 'fresh_document_pilot' or receipt['model_revision'] != config['model_revision']):
                raise ValueError('The angle study does not match the grid archive and model')
            traces = {}
            for seed in seeds:
                payload = archive.read(f'seed_{seed}/angle_screen.json')
                if hashlib.sha256(payload).hexdigest() != receipt['by_seed'][str(seed)]['trace_sha256']:
                    raise ValueError('Angle selection checksum differs')
                traces[seed] = json.loads(payload)
    else:
        grids, traces = fit_7b['grids'], fit_7b['traces']
    records, raw, placements, mean_nmse = [], {}, None, {}
    seen_docs = {d for r in grids.values() for key in ('fit_documents','heldout_documents') for d in r[key]}
    for seed in seeds:
        ready, ref = prepared['seeds'][seed], grids[seed]
        panels = ready['heldout']
        dev = read_json(config['data']['token_manifests'][str(seed)])['partitions']['dev']
        ids, blocks = [d['document_id'] for d in dev], [d['block_id'] for d in dev]
        current = {p.panel_id:p.placement for p in panels}
        if ('fit' in ready or len(current) != 9 or len(panels) != 9 or (placements is not None and current != placements)
                or len(ids) != design['documents_per_seed'] or len(set(ids)) != len(ids) or seen_docs.intersection(ids)
                or any(p.metadata['source_type'] != 'real_layer' or p.metadata['document_ids'] != ids
                       or p.metadata['block_ids'] != blocks or p.metadata['tokens_per_block'] != [design['tokens_per_document']]*len(ids)
                       for p in panels)):
            raise ValueError('Paired evaluation requires the nine matching panels on the frozen fresh article list')
        seen_docs.update(ids)
        placements = current
        if fit_7b is None:
            warm = refs[seed][WARM]['solver']
            grid = replay_continued_grid(refs[seed], dimension, seed, device)
        else:
            if (current != {p['panel_id']:p['placement'] for p in ref['heldout']}
                    or any(p.metadata['graph_identity'] != ref['graph_identity'] for p in panels)):
                raise ValueError('Captured graph or projection identities differ from the fitted A100 grid')
            warm = ref['solver']['grid_stages'][0]
            grid = replay_retest_parent(ref, device)
        prefix, _ = prefix_rotation(dimension, seed, warm, device)
        learned = replay_angle_arms(prefix, traces[seed])
        panel_sha = panel_fingerprint(panels)
        if document_reference is not None:
            row_seeds[seed] = score_seed(seed, panels, panel_sha, full_up_weight, {CONTINUED:grid, PREFIX:prefix, new:learned[new]},
                                         ready['specs'], document_rows, progress=progress)
            path = atomic_json(Path(output)/f'seed_{seed}'/'row_diagnostic.json', row_seeds[seed])
            raw[path.relative_to(output).as_posix()] = sha256_file(path)
            del grid, prefix, learned
            continue
        rotations = {CONTINUED:grid, HAD:hadamard_rotation(dimension, seed, device), **learned}
        mean_nmse[str(seed)] = {}
        for method, q in rotations.items():
            report(progress, 'bilateral.fresh_documents', seed=seed, method=method, documents=len(ids))
            measured = panel_metrics(panels, q, ready['specs'], device=device, error_terms=fit_7b is not None)
            mean_nmse[str(seed)][method] = float(np.mean([p['nmse'] for p in measured]))
            orth = float(torch.linalg.norm(q.T@q - torch.eye(dimension, dtype=q.dtype, device=q.device))/dimension**.5)
            row = {'seed':seed, 'method':method, 'q_sha256':q_digest(q), 'source_type':'real_layer', 'heldout':measured,
                   'heldout_panel_sha256':panel_sha, 'orthogonality_residual':orth, 'heldout_documents':ids}
            path = atomic_json(Path(output)/f'seed_{seed}'/method/'document_result.json', row)
            raw[path.relative_to(output).as_posix()] = sha256_file(path)
            records.append({'seed':seed, 'method':method, 'panels':{p['panel_id']:p['blocks'] for p in measured}})
        del rotations, prefix, grid, q, learned
    if document_reference is not None:
        return {'status':'DIAGNOSTIC_COMPLETE','source_type':'real_layer','evidence_kind':'FROZEN_ROW_DIAGNOSTIC',
                'site':SITE,'scope':'post-hoc row and precision ablation on the already evaluated articles',
                'document_reference_sha256':sha256_file(document_reference),'fit_archive_sha256':sha256_file(fit_archive),
                'reference_archive_sha256':sha256_file(grid_archive),'bootstrap_draws':design['bootstrap_draws'],
                'bootstrap_seed':design['bootstrap_seed'],'tile_rows':TILE_ROWS,'row_artifacts':raw,
                'row_counts_by_seed':{s:r['methods'][new]['row_counts'] for s,r in row_seeds.items()},
                'row_comparisons':summarize_rows(row_seeds, new, design)}
    stats = paired_site_gains(records, seeds, placements, {'vs_output_only': (new, original), 'vs_full_grid': (new, CONTINUED),
                              'vs_hadamard': (new, HAD)}, repetitions=design['bootstrap_draws'], rng_seed=design['bootstrap_seed'],
                              watched_sites=[design['watch_site']] if fit_7b is not None else ())
    if fit_7b is not None:
        stats['mean_site_nmse_by_seed'] = mean_nmse
    return {'status': 'STUDY_COMPLETE', 'evidence_kind': 'FROZEN_BILATERAL_DOCUMENT_STUDY', 'source_type': 'real_layer',
            'reference_archive_sha256': sha256_file(grid_archive), 'fit_archive_sha256': sha256_file(fit_archive),
            'design': design, 'document_artifacts': raw, 'document_summary': stats}
