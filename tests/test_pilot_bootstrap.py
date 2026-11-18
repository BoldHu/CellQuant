import copy

import numpy as np
import pytest

from cellquant.baselines import METHODS
from cellquant.pilot_bootstrap import CONTROLS, paired_document_intervals


def example():
    seeds, sites = [11, 23], ['down', 'up', 'q']
    records, errors, targets = [], {}, {}
    for i, seed in enumerate(seeds):
        for method_id, method in enumerate(METHODS):
            panels = {}
            for j, site in enumerate(sites):
                rows = []
                for k, (doc, block) in enumerate([('a', 'a0'), ('a', 'a1'), ('b', 'b0'), ('c', 'c0')]):
                    energy = 3 + k * 4 + j
                    error = 1 + (method_id + i + j * k) % 4 + .4 * k
                    rows.append(dict(document_id=doc, block_id=block, sampled_tokens=2,
                                     squared_error=error, target_energy=energy))
                    key = (method, seed, site, doc)
                    errors[key] = errors.get(key, 0) + error
                    targets[key] = targets.get(key, 0) + energy
                panels[site] = rows[::-1] if method_id % 2 else rows
            records.append(dict(seed=seed, method=method, panels=panels))

    def matrices(counts):
        def metric(m, s, p, values):
            return sum(n * values[m, s, p, d] for d, n in zip('abc', counts))
        return {m: np.array([[metric(m,s,p,errors) / metric(m,s,p,targets) for p in sites]
                             for s in seeds]) for m in METHODS}

    raw = matrices([1, 1, 1])
    frozen = dict(seeds=seeds, sites=sites, new_nmse=raw['cellquant'], hadamard_nmse=raw['hadamard_shared'],
                  best_control_nmse=np.minimum.reduce([raw[m] for m in CONTROLS]))
    return records, frozen, matrices


def test_document_resampling_pairs_across_seeds_and_reselects_control_on_each_draw():
    rows, frozen, matrices = example()
    result = paired_document_intervals(rows, frozen, repetitions=100, seed=12)
    assert result['document_strata'] == [{'seeds': [11, 23], 'documents': 3}]
    # Three shared documents, including one containing two dependent blocks.
    draws = []
    for weights in np.random.default_rng(12).multinomial(3, [1/3]*3, size=100):
        values = matrices(weights)
        oracle = np.minimum.reduce([values[m] for m in CONTROLS])
        draws.append(np.median((100 * (1 - values['cellquant'] / oracle)).mean(axis=0)))
    np.testing.assert_allclose(result['comparisons']['best_matched_per_site']['percentile_95_pct'],
                               np.percentile(draws, [2.5, 97.5]))


@pytest.mark.parametrize('issue,reason', [('document','Unpaired'), ('duplicate','duplicate'),
                         ('method','one record'), ('energy','target energy'), ('matrix','frozen G1')])
def test_mismatched_or_relabelled_results_are_rejected(issue, reason):
    records, frozen, _ = example()
    block = records[0]['panels']['down'][0]
    if issue == 'document':
        block['document_id'] = 'other'
    elif issue == 'duplicate':
        records[0]['panels']['down'].append(copy.deepcopy(block))
    elif issue == 'method':
        records.pop()
    elif issue == 'energy':
        block['target_energy'] *= 2
    else:
        frozen['new_nmse'][0, 0] *= 2
    with pytest.raises(ValueError, match=reason):
        paired_document_intervals(records, frozen, repetitions=2)
