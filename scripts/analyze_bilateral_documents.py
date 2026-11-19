"""Recompute the frozen A100 document decision and decompose the watched site's regression."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from cellquant.bilateral_documents import COHORT_FILE, DESIGN, FIT_FILE, GRID_FILE, block_identity, checked_json, document_decision
from cellquant.bilateral_screen import DESIGN as ANGLES
from cellquant.data import token_hash
from cellquant.matching_study import CONTINUED, HAD, attribution, paired_site_gains
from cellquant.runtime import atomic_json, config_hash, read_json, sha256_file


def compare(actual, expected):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError('Keys differ between recomputation and archived results')
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Array size differs from archived result')
        for item, other in zip(actual, expected):
            compare(item, other)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    elif actual != expected:
        raise ValueError(f'Recomputed value {actual!r} differs from archived {expected!r}')


def audit(path, refs):
    old, new = ANGLES['arms']
    seeds, site = ANGLES['seeds'], DESIGN['watch_site']
    fit = read_json(refs/'bilateral_fit_7b_summary.json')
    cohort = read_json(refs/COHORT_FILE)
    placements, records, means, diagnosis = {}, [], {}, {}
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Invalid ZIP CRC')
        receipt = json.loads(archive.read('run/result.json'))
        config = json.loads(archive.read('run/config.json'))
        protocol = checked_json(archive, 'inputs/protocol.json', receipt['protocol_sha256'])
        if (receipt['status'] != 'STUDY_COMPLETE' or receipt['source_type'] != 'real_layer'
                or receipt['execution_mode'] != 'frozen_bilateral_document_pilot' or receipt['design'] != DESIGN
                or config_hash(config) != receipt['config_sha256'] or receipt['fit_source'] != protocol['bilateral_document_pilot']
                or receipt['fit_archive_sha256'] != sha256_file(refs/FIT_FILE)
                or receipt['reference_archive_sha256'] != sha256_file(refs/GRID_FILE)):
            raise ValueError('Wrong run, protocol or fitted rotations')
        for name, digest in receipt['fit_source']['archives'].items():
            contents = archive.read('inputs/fit_reference/' + name)
            matches_ref = json.loads(contents) == cohort if name == COHORT_FILE else digest == sha256_file(refs/name)
            if hashlib.sha256(contents).hexdigest() != digest or not matches_ref:
                raise ValueError(f'Frozen input differs: {name}')
        with ZipFile(refs/FIT_FILE) as screen:
            prior = [*cohort['previous_42m_train_blocks'], *json.loads(screen.read('inputs/prior_documents.json'))]
            for seed in seeds:
                parts = json.loads(screen.read(f'inputs/seed_{seed}/data_manifest.json'))['partitions']
                prior.extend(b for rows in parts.values() for b in rows)
        ignored = {k:{b[k] for b in prior} for k in block_identity(prior[0])}
        fresh = [b for rows in cohort['heldout'].values() for b in rows]
        for key, used in ignored.items():
            values = {b[key] for b in fresh}
            if len(values) != len(seeds)*DESIGN['documents_per_seed'] or values & used:
                raise ValueError(f'Repeated article or block in scored cohort: {key}')
        expected = {f'seed_{s}/{m}/document_result.json' for s in seeds for m in (old, new, CONTINUED, HAD)}
        if set(receipt['document_artifacts']) != expected:
            raise ValueError('Incomplete seed or method set')
        for seed in seeds:
            manifest = checked_json(archive, f'inputs/seed_{seed}/data_manifest.json', protocol['token_manifest_sha256'][str(seed)])
            parts = manifest['partitions']
            compare([block_identity(b) for b in parts['dev']], cohort['heldout'][str(seed)])
            token_data = archive.read(f'inputs/seed_{seed}/token_ids.npz')
            if hashlib.sha256(token_data).hexdigest() != manifest['tokens_sha256']:
                raise ValueError('Wrong token archive')
            with np.load(io.BytesIO(token_data), allow_pickle=False) as arrays:
                for blocks in parts.values():
                    for b in blocks:
                        if token_hash(arrays[b['array_key']]) != b['token_ids_sha256']:
                            raise ValueError('Token sequence differs from manifest')
            ids = [b['document_id'] for b in parts['dev']]
            per_method, means[str(seed)], panel_hashes = {}, {}, set()
            for method in (old, new, CONTINUED, HAD):
                name = f'seed_{seed}/{method}/document_result.json'
                row = checked_json(archive, 'run/'+name, receipt['document_artifacts'][name])
                if (row['source_type'] != 'real_layer' or row['seed'] != seed or row['method'] != method
                        or row['heldout_documents'] != ids or (method != HAD and
                            row['q_sha256'] != fit['by_seed'][str(seed)]['replayed_q_sha256'][method])):
                    raise ValueError(f'Unmatched measurements in {name}')
                panel_hashes.add(row['heldout_panel_sha256'])
                panels = row['heldout']
                current = {p['panel_id']:p['placement'] for p in panels}
                if len(panels) != 9 or len(current) != 9 or (placements and current != placements):
                    raise ValueError('Panel identities differ')
                placements = current
                for p in panels:
                    blocks = p['blocks']
                    compare([(b['document_id'], b['block_id'], b['sampled_tokens']) for b in blocks],
                            [(b['document_id'], b['block_id'], DESIGN['tokens_per_document']) for b in parts['dev']])
                    for b in blocks:
                        compare(sum(b[k] for k in ('weight_error_energy','activation_error_energy','cross_energy')), b['squared_error'])
                    compare(sum(b['squared_error'] for b in blocks)/sum(b['target_energy'] for b in blocks), p['nmse'])
                means[str(seed)][method] = float(np.mean([p['nmse'] for p in panels]))
                records.append({'seed':seed,'method':method,'panels':{p['panel_id']:p['blocks'] for p in panels}})
                per_method[method] = next(p for p in panels if p['panel_id'] == site)
            if len(panel_hashes) != 1:
                raise ValueError('Methods were scored on different panels')
            base, candidate = per_method[CONTINUED], per_method[new]
            before, after = (attribution([p])[site] for p in (base, candidate))
            penalty = {k:100*(after[k]-before[k])/base['nmse'] for k in ('weight_error_energy','activation_error_energy','cross_energy')}
            compare(sum(penalty.values()), 100*(candidate['nmse']/base['nmse']-1))
            diagnosis[str(seed)] = {'penalty_components_pct_of_grid_nmse':penalty, 'documents_with_higher_sse':sum(
                a['squared_error'] > b['squared_error'] for a,b in zip(candidate['blocks'],base['blocks'])),
                'document_count':len(ids), 'watched_site_nmse':{m:p['nmse'] for m,p in per_method.items()}}
        code = json.loads(archive.read('run/execution_fingerprint.json'))['payload']['code_git']
    stats = paired_site_gains(records, seeds, placements, {'vs_output_only': (new,old), 'vs_full_grid': (new,CONTINUED),
                              'vs_hadamard': (new,HAD)}, repetitions=DESIGN['bootstrap_draws'], rng_seed=DESIGN['bootstrap_seed'],
                              watched_sites=[site])
    stats['mean_site_nmse_by_seed'] = means
    compare(stats, receipt['document_summary'])
    decision = document_decision(stats)
    compare(decision, receipt['pilot_decision'])
    compare(decision['next_step'], receipt['next_step'])
    return {'run_id':receipt['run_id'], 'code_git':code, 'source_type':'real_layer', 'archive_sha256':sha256_file(path),
            'model_id':config['formal_pilot']['model_id'], 'model_revision':config['model_revision'], 'design':DESIGN,
            'excluded_prior_documents':len(ignored['document_id']),
            'document_summary':stats, 'pilot_decision':decision, 'watched_site_attribution':diagnosis}


if __name__ == '__main__':
    refs = Path(__file__).resolve().parents[1]/'evidence/output_matching_20261119'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, nargs='?', default=refs/'bilateral_documents_7b.zip')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args.archive, refs)
    print(atomic_json(args.output, summary), summary['pilot_decision']['next_step'])
