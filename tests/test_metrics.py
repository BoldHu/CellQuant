import math
import numpy as np
import pytest

from cellquant.metrics import ppl_from_nll, paired_nll_bootstrap


def test_weighted_nll_not_mean_of_batch_ppl():
    result = ppl_from_nll([1.0, 18.0], [1, 9])
    assert result["ppl"] == pytest.approx(math.exp(1.9))
    assert result["ppl"] != pytest.approx((math.e + math.exp(2)) / 2)


def test_paired_bootstrap_constant_known_effect():
    counts = np.array([10, 20, 30, 40])
    base = np.ones((3, 4)) * counts * 2.0
    new = base - counts * 0.02
    r = paired_nll_bootstrap(base, new, counts, ["a", "a", "b", "c"], repetitions=200)
    assert r["delta_new_minus_base"] == pytest.approx(-0.02)
    assert r["ci95_high"] == pytest.approx(-0.02)
    assert r["document_clusters"] == 3
    assert r["ppl_gain_pct"] == pytest.approx(100 * (1 - math.exp(-0.02)))


@pytest.mark.parametrize("counts", [[0], [-1], [float('nan')], [1.5]])
def test_invalid_counts_do_not_produce_results(counts):
    with pytest.raises(ValueError):
        ppl_from_nll([1.0], counts)


def test_single_document_is_not_independent_bootstrap_evidence():
    with pytest.raises(ValueError):
        paired_nll_bootstrap([[2., 4.]], [[2., 4.]], [1, 2], ["same", "same"])
