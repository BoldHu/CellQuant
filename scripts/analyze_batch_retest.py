"""Recompute the document contrasts and diagnose a parent-bound stop from its portable archive."""
import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from cellquant import batch_retest, batch_validation
from cellquant.data import token_hash
from cellquant.gates import load_rules, resolve_rules_path
from cellquant.matching_study import CONTINUED, HAD, paired_site_gains
from cellquant.output_matching import fits_limits
from cellquant.pilot import pilot_solver_config
from cellquant.runtime import atomic_json, config_hash, sha256_file


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'evidence/output_matching_20261119/batch_documents_7b.zip'


def audit(bundle, pilot=PILOT):
    digest = lambda data: hashlib.sha256(data).hexdigest()
    candidate, random = batch_retest.CANDIDATE, batch_validation.RANDOM
    prior, _, excluded = batch_retest.read_document_pilot(pilot)
    with ZipFile(bundle) as archive:
        def read(path):
            return json.loads(archive.read(path))

        def require(condition, message):
            if not condition:
                raise ValueError(message)

        require(archive.testzip() is None, 'Invalid archive CRC')
        result, protocol = read('run/result.json'), read('inputs/protocol.json')
        config, spec = read('run/config.json'), protocol['batch_g1_retest']
        require(result['status'] == 'RETEST_STOPPED_BY_BOUND' and result['source_type'] == 'real_layer'
                and not result['complete_comparison'] and 'run/G1.json' not in archive.namelist(),
                'Expected the incomplete, bound-stopped retest')
        require(config == read('inputs/config.json') and config_hash(config) == result['config_sha256']
                and config['batch_g1_retest'] == spec and spec['design'] == batch_retest.DESIGN
                and spec['pilot_sha256'] == sha256_file(pilot) and prior['run_id'] == spec['pilot_run_id']
                and protocol['status'] == 'FROZEN'
                and digest(archive.read('inputs/protocol.json')) == read('inputs/execution_provenance.json')['protocol_sha256']
                and spec['gate_rules_sha256'] in {sha256_file(resolve_rules_path()), digest(Path(resolve_rules_path()).read_bytes().replace(b'\r\n', b'\n'))},
                'Retest protocol or gate rules differ')
        require(read('inputs/prior_documents.json') == excluded
                and digest(archive.read('inputs/prior_documents.json')) == spec['excluded_sha256'], 'Earlier cohort differs')
        fields = ('document_id', 'document_sha256', 'token_ids_sha256')
        used = {k: {b[k] for b in excluded} for k in fields}
        rows, blocks, hashes, diagnostics = [], [], {}, {}
        placements = None
        seeds = spec['design']['seeds']
        tolerance = pilot_solver_config(config).budget_tolerance
        max_nmse_error = 0.
        for seed in seeds:
            base = f'inputs/seed_{seed}/'
            manifest = read(base + 'data_manifest.json')
            manifest_hash = digest(archive.read(base + 'data_manifest.json'))
            cache = archive.read(base + 'token_ids.npz')
            require(manifest_hash == protocol['token_manifest_sha256'][str(seed)] and digest(cache) == manifest['tokens_sha256'],
                    'Token manifest or cache checksum mismatch')
            split = manifest['partitions']
            with np.load(BytesIO(cache), allow_pickle=False) as tokens:
                for role in ('fit', 'dev'):
                    require(len(split[role]) == spec['design'][role], 'Unexpected cohort size')
                    for b in split[role]:
                        require(token_hash(tokens[b['array_key']]) == b['token_ids_sha256'], 'Block hash mismatch')
                        for k in fields:
                            require(b[k] not in used[k], f'Reused {k}')
                            used[k].add(b[k])
            arms = {}
            for method in (HAD, CONTINUED, candidate, random):
                filename = 'random_result' if method == random else 'pilot_result'
                path = f'run/seed_{seed}/{method}/{filename}.json'
                row = read(path)
                hashes[path] = digest(archive.read(path))
                require(row['status'] == 'COMPLETE' and row['method'] == method and row['seed'] == seed
                        and row['evidence_kind'] == 'FORMAL_BATCH_G1' and row['data_manifest_sha256'] == manifest_hash
                        and row['fit_documents'] == sorted(b['document_id'] for b in split['fit'])
                        and row['heldout_documents'] == sorted(b['document_id'] for b in split['dev']),
                        'Method record or document role mismatch')
                expected = {(b['document_id'], b['block_id']) for b in split['dev']}
                sites = {p['panel_id']: p['placement'] for p in row['heldout']}
                require(len(row['heldout']) == len(sites) == 9 and placements in (None, sites), 'Panel set differs')
                placements = sites
                for p in row['heldout']:
                    measured = p['blocks']
                    require(len(measured) == len(expected) and {(b['document_id'], b['block_id']) for b in measured} == expected
                            and all(b['sampled_tokens'] == 32 for b in measured), 'Held-out block pairing differs')
                    nmse = sum(b['squared_error'] for b in measured) / sum(b['target_energy'] for b in measured)
                    max_nmse_error = max(max_nmse_error, abs(nmse - p['nmse']))
                rows.append(row)
                blocks.append({'seed': seed, 'method': method, 'panels': {p['panel_id']: p['blocks'] for p in row['heldout']}})
                arms[method] = row
            parent, new = arms[CONTINUED], arms[candidate]
            for k in ('fit_panel_fingerprint', 'heldout_panel_fingerprint', 'hardware_sha256', 'graph_identity', 'schedule_sha256'):
                require(len({r[k] for r in arms.values()}) == 1, f'Unmatched {k}')
            selection, proposal = new['solver']['selection'], new['solver']['proposal']
            trace = selection['trace']
            require([t['batch'] for t in trace if t['accepted']] == selection['accepted_batches']
                    and len(trace) == selection['batch_count'] == 32, 'Batch decisions differ')
            for trial in trace:
                feasible = fits_limits(trial, selection['limits'], tolerance)
                require(trial['feasible'] == feasible and trial['accepted'] == (feasible and trial['loss'] < trial['parent_loss']),
                        'Recorded fit decision disagrees with the measured risks or joint loss')
            inputs = [(t['loss'] - t['parent_loss']) - (t['output_loss'] - t['parent_output_loss']) for t in trace]
            outputs = [t['output_loss'] - t['parent_output_loss'] for t in trace]
            ratios = [x / -y if y < 0 else None for x, y in zip(inputs, outputs)]
            full = proposal['joint_fit_trials'][-1]
            total_change = full['loss'] - proposal['initial']['loss']
            output_change = full['output_loss'] - proposal['initial']['output_loss']
            w, short = parent['solver']['grid_stages']
            diagnostics[str(seed)] = {
                'nonzero_proposed_angles': sum(a != 0 for a in proposal['proposed_angles']),
                'batches': len(trace), 'within_both_risk_caps': sum(t['feasible'] for t in trace),
                'joint_loss_lower': sum(t['loss'] < t['parent_loss'] for t in trace),
                'output_loss_lower': sum(d < 0 for d in outputs), 'input_loss_higher': sum(d > 0 for d in inputs),
                'input_penalty_per_unit_output_gain_by_batch': ratios,
                'median_input_penalty_per_unit_output_gain': float(np.median(ratios)) if all(x is not None for x in ratios) else None,
                'accepted_batches': selection['accepted_batches'], 'applied_nonzero_angles': selection['applied_pairs'],
                'best_batch_joint_delta': min(t['loss'] - t['parent_loss'] for t in trace),
                'full_proposal_delta_fit': {'input': total_change - output_change, 'output': output_change, 'joint': total_change},
                'candidate_grid_random_record_same_q_file_hash': len({arms[m]['q_sha256'] for m in (CONTINUED, candidate, random)}) == 1,
                'final_q_matches_grid_array_hash': selection['final_q_sha256'] == parent['parent_q_sha256'],
                'random_selected_batches': selection['same_count_random']['batches'],
                'full_grid_seconds': parent['total_calibration_seconds'], 'candidate_seconds': new['total_calibration_seconds'],
                'extra_seconds': new['total_calibration_seconds'] - parent['total_calibration_seconds'],
                'grid_128_seconds_including_capture': parent['total_calibration_seconds'] - short['seconds'],
                'fit_loss_grid_128': w['final']['loss'], 'fit_loss_grid_256': short['final']['loss']}
        require(max_nmse_error < 1e-12, 'NMSE differs from its block sums')
        contrasts = {'selected_vs_grid': (candidate, CONTINUED), 'random_vs_grid': (random, CONTINUED),
                     'selected_vs_random': (candidate, random), 'selected_vs_hadamard': (candidate, HAD)}
        stats = paired_site_gains(blocks, seeds, placements, contrasts,
                                  repetitions=batch_validation.DESIGN['bootstrap_draws'], rng_seed=batch_validation.DESIGN['bootstrap_seed'])
        saved = read('run/paired_parent_and_random.json')
        require(all(stats['comparisons'][k] == v for k, v in saved['comparisons'].items()), 'Paired comparisons differ')
        bounds = batch_retest.route_bounds(rows, load_rules())
        require(bounds == result['route_bounds'] == read('run/route_bounds.json') and bounds['both_routes_ruled_out'],
                'Screening upper bounds differ')
        fp = read('run/execution_fingerprint.json')
        return {'source_type': result['source_type'], 'run_id': result['run_id'], 'status': result['status'],
                'next_step': result['next_step'], 'complete_comparison': result['complete_comparison'],
                'archive_sha256': sha256_file(bundle), 'pilot_sha256': spec['pilot_sha256'], 'method_file_sha256': hashes,
                'model_id': protocol['model_id'], 'model_revision': protocol['model_revision'],
                'hardware': fp['payload']['environment']['cuda']['devices'][0]['name'],
                'git': fp['payload']['code_git'], 'fresh_fit_articles': len(seeds) * spec['design']['fit'],
                'fresh_heldout_articles': stats['distinct_documents'], 'max_nmse_reconstruction_error': max_nmse_error,
                'recorded_dense_q_files_in_zip': any(n.endswith('Q.npz') for n in archive.namelist()),
                'route_bounds': bounds, 'by_seed': diagnostics, 'document_contrasts': stats}


def plot_batch_costs(summary, destination):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, ax = plt.subplots(figsize=(8.1, 3.4), layout='constrained')
    for seed, offset, color, marker in [('11', -.13, '#17647B', 'o'), ('23', .13, '#B44C30', '^')]:
        ratios = summary['by_seed'][seed]['input_penalty_per_unit_output_gain_by_batch']
        if any(r is None for r in ratios):
            raise ValueError('A cost ratio requires a positive output gain for every plotted batch')
        ax.scatter(np.arange(1, len(ratios)+1)+offset, ratios, s=25, color=color, marker=marker, label=f'Seed {seed}')
    ax.axhline(1, color='#4D5358', linewidth=1, linestyle='--', label='No change in joint loss')
    ax.grid(axis='y', color='#DBDDE1', linewidth=.6)
    ax.set_axisbelow(True)
    ax.set(xlabel='Candidate batch (64 rotation planes)', ylabel='Input loss increase / output loss reduction',
           title='LLaMA-2-7B calibration after a 256-pair grid search', ylim=(0, None))
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=3, frameon=False)
    fig.savefig(destination, dpi=190, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    summary = audit(args.bundle)
    if args.output:
        atomic_json(args.output, summary)
    if args.figure:
        plot_batch_costs(summary, args.figure)
    print(json.dumps({'status': summary['status'], 'bounds': summary['route_bounds'],
                      'seeds': summary['by_seed'], 'hadamard': summary['document_contrasts']['comparisons']['selected_vs_hadamard']['all']}, indent=2))
