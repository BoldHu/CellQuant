"""Exploratory paired-document uncertainty for the frozen G1 NMSE statistic."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .baselines import METHODS
from .runtime import atomic_json, read_json, sha256_file


CONTROLS = ('angle_grid_matched', 'fixed_scale_search_matched', 'givens_ste_matched')


def _document_data(records, seeds, sites):
    by_method = {(r['seed'], r['method']): r['panels'] for r in records}
    if len(by_method) != len(records) or set(by_method) != {(s, m) for s in seeds for m in METHODS}:
        raise ValueError('Expected one record per seed and method')
    totals = {}
    membership = defaultdict(set)
    for seed in seeds:
        reference = None
        for method in METHODS:
            panels = by_method[seed, method]
            if set(panels) != set(sites):
                raise ValueError('Unmatched panel IDs')
            for site in sites:
                identity = {}
                by_doc = defaultdict(lambda: np.zeros(2))
                for block in panels[site]:
                    doc, bid = block['document_id'], block['block_id']
                    count = block['sampled_tokens']
                    if not doc or not bid or bid in identity or type(count) is not int or count < 1:
                        raise ValueError('Missing or duplicate block identity, or invalid token count')
                    identity[bid] = (doc, count)
                    values = np.array([block['squared_error'], block['target_energy']], dtype=float)
                    if not np.isfinite(values).all() or (values < 0).any():
                        raise ValueError('Invalid squared error or target energy')
                    by_doc[doc] += values
                    membership[doc].add(seed)
                if reference is None:
                    reference = identity
                if not identity or identity != reference:
                    raise ValueError('Unpaired evaluation documents, blocks or sampled token counts')
                totals[seed, method, site] = by_doc
    documents = sorted(membership)
    doc_index = {doc: i for i, doc in enumerate(documents)}
    stats = np.zeros((len(METHODS), len(seeds), len(sites), len(documents), 2))
    for mi, method in enumerate(METHODS):
        for si, seed in enumerate(seeds):
            for pi, site in enumerate(sites):
                for doc, values in totals[seed, method, site].items():
                    stats[mi, si, pi, doc_index[doc]] = values
    if not np.allclose(stats[..., 1], stats[METHODS.index('hadamard_shared'), ..., 1],
                       rtol=1e-10, atol=1e-15):
        raise ValueError('Paired methods disagree on document target energy')
    strata = defaultdict(list)
    for doc in documents:
        strata[tuple(sorted(membership[doc]))].append(doc_index[doc])
    return stats, strata, documents


def _gains(weights, stats):
    # A document has one multiplicity across all its methods, sites and seeds.
    sums = np.einsum('bd,msldk->bmslk', weights, stats, optimize=True)
    if (sums[..., 1] <= 0).any():
        raise ValueError('Zero target energy in a document resample')
    nmse = sums[..., 0] / sums[..., 1]
    comparator = {m: nmse[:, METHODS.index(m)] for m in ('hadamard_shared', *CONTROLS)}
    comparator['best_matched_per_site'] = np.minimum.reduce([comparator[m] for m in CONTROLS])
    if any((v <= 0).any() for v in comparator.values()):
        raise ValueError('Zero comparator NMSE in a document resample')
    cell = nmse[:, METHODS.index('cellquant')]
    gains = {name: (100 * (1 - cell / other)).mean(axis=1) for name, other in comparator.items()}
    return gains, nmse


def paired_document_intervals(records, distributions, *, repetitions=10000, seed=20261115):
    seeds, sites = distributions['seeds'], distributions['sites']
    if (repetitions < 2 or not seeds or len(set(seeds)) != len(seeds)
            or not sites or len(set(sites)) != len(sites)):
        raise ValueError('Invalid repetitions, seeds or sites')
    stats, strata, docs = _document_data(records, seeds, sites)
    point, nmse = _gains(np.ones((1, len(docs))), stats)
    expected = (('cellquant', 'new_nmse'), ('hadamard_shared', 'hadamard_nmse'))
    matrices = [(nmse[0, METHODS.index(m)], k) for m, k in expected]
    matrices.append((np.minimum.reduce([nmse[0, METHODS.index(m)] for m in CONTROLS]), 'best_control_nmse'))
    for values, key in matrices:
        ref = np.asarray(distributions[key], dtype=float)
        if values.shape != ref.shape or not np.allclose(values, ref, rtol=1e-10, atol=1e-15):
            raise ValueError(f'Block energy sums disagree with the frozen G1 matrix {key}')

    rng = np.random.default_rng(seed)
    weights = np.zeros((repetitions, len(docs)), dtype=int)
    for ids in strata.values():
        n = len(ids)
        weights[:, ids] = rng.multinomial(n, np.full(n, 1/n), size=repetitions)
    draws = defaultdict(list)
    for start in range(0, repetitions, 256):
        gains, _ = _gains(weights[start:start + 256], stats)
        for name, values in gains.items():
            draws[name].append(np.median(values, axis=1))
    contrasts = {}
    for name, gain in point.items():
        interval = np.percentile(np.concatenate(draws[name]), [2.5, 97.5])
        contrasts[name] = {'median_site_gain_pct': float(np.median(gain[0])), 'percentile_95_pct': interval.tolist()}
    return {'evidence_kind': 'POSTHOC_PAIRED_DOCUMENT_DIAGNOSTIC', 'seeds_held_fixed': seeds, 'sites': sites,
            'repetitions': repetitions, 'bootstrap_seed': seed, 'distinct_documents': len(docs),
            'document_strata': [{'seeds': list(group), 'documents': len(ids)} for group, ids in strata.items()],
            'scope': 'Percentile intervals conditional on fitted rotations and sampled sites. Refit no models. '
                     'Resample whole documents within seed-membership strata, paired across methods/sites. '
                     'Re-select the minimum of the three matched controls per seed/site in each draw.',
            'comparisons': contrasts}


def analyze_run(run_dir):
    root = Path(run_dir)
    evidence_path = root / 'G1_evidence.json'
    evidence = read_json(evidence_path)
    if (evidence.get('gate_id') != 'G1' or evidence.get('source_type') != 'real_layer'
            or not evidence.get('provenance', {}).get('complete_comparison')):
        raise ValueError('A completed real-layer G1 comparison is required')
    digest = sha256_file(evidence_path)
    if any(root.glob('seed_*/*/pilot_result.json')):
        records = []
        for ref in evidence['artifact_refs']:
            # Resolve within this restored run, even if Colab absolute paths have changed.
            tail = Path(ref['path'].replace('\\', '/')).parts[-3:]
            if len(tail) != 3 or tail[1] not in METHODS or tail[2] != 'pilot_result.json':
                raise ValueError('Unexpected G1 method artifact path')
            path = root.joinpath(*tail)
            if not tail[0].startswith('seed_') or sha256_file(path) != ref['sha256']:
                raise ValueError('Unverified method result')
            row = read_json(path)
            if (row['source_type'] != 'real_layer' or row['status'] != 'COMPLETE'
                    or row['method'] != tail[1] or tail[0] != f"seed_{row['seed']}"):
                raise ValueError('Method result identity or status mismatch')
            records.append({'seed': row['seed'], 'method': row['method'],
                            'panels': {p['panel_id']: p['blocks'] for p in row['heldout']}})
        source = {'kind': 'verified_method_artifacts', 'count': len(records)}
    else:
        path = root / 'heldout_blocks.json'
        extracted = read_json(path)
        if extracted.get('source_type') != 'real_layer' or extracted.get('g1_evidence_sha256') != digest:
            raise ValueError('Document extracts do not match this G1 evidence')
        records = extracted['records']
        source = {'kind': 'notebook_block_field_extracts', 'sha256': sha256_file(path)}
    report = paired_document_intervals(records, read_json(root / 'G1_distributions.json'))
    return dict(report, source_type='real_layer', evidence_sha256=digest, block_source=source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = analyze_run(args.run_dir)
    atomic_json(args.output, result)
    print(result['comparisons'])
