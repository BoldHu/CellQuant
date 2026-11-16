"""Meaningful reference-oracle tests; no mocked experiment success."""
import math
import numpy as np
import pytest

from cellquant.math_reference import (
    PairProblem, quantize_absmax, givens, enumerate_events,
    analytic_cell_coefficients, trig_value, solve_pair_reference,
    run_reference_checks,
)


def test_complete_groups_and_quantizer_ties():
    _, codes, _ = quantize_absmax([[7., .5, 1.5, 2.5, -.5, -1.5, -2.5, -7.]])
    assert np.array_equal(codes, [[7,0,2,2,0,-2,-2,-7]])
    zero, code, scale = quantize_absmax(np.zeros((3,16)), group_size=8)
    assert np.isfinite(scale).all() and np.count_nonzero(zero) == np.count_nonzero(code) == 0
    with pytest.raises(ValueError, match='divide'):
        quantize_absmax(np.ones((3,15)), group_size=8)


def test_screening_equality_and_reference_guard():
    # M=r=5 is safe, even where the maximum owner ties.
    p = PairProblem(np.array([[3.,4.,5.,0.]]),np.array([[3.],[4.],[5.],[0.]]),(0,1))
    assert p.stable.all()
    for theta in np.linspace(-math.pi,math.pi,101):
        assert np.allclose(np.max(np.abs(p.values(theta)),axis=1),5.,atol=1e-12,rtol=0)
    below = np.nextafter(5.,0.)
    q = PairProblem(np.array([[3.,4.,below,0.]]),np.array([[3.],[4.],[below],[0.]]),(0,1))
    assert not q.stable.any()  # No tolerance may falsely certify a changing scale.
    wide = PairProblem(np.ones((1,512)),np.ones((512,1)),(0,1))
    with pytest.raises(ValueError,match='Do not truncate'):
        enumerate_events(wide)


def test_isolated_owner_tie_at_cell_midpoint_uses_constant_scale():
    # At theta=0, argmax picks rotating coordinate 0 rather than constant
    # coordinate 2. The true scale is nevertheless 5/7 throughout the orbit.
    W = np.array([[5., 0., 5., 2.]])
    p = PairProblem(W, W.T, (0, 1))
    assert p.stable.all()
    full, _ = enumerate_events(p, False)
    screened, _ = enumerate_events(p, True)
    assert np.array_equal(full, screened)
    # Check the midpoint tie itself, and arbitrary interior values on both
    # sides. This catches a real missing-code-event bug, not just owner labels.
    product, loss, bw, ba = analytic_cell_coefficients(p, 0.)
    for theta in (-.01, 0., .01):
        actual = p.evaluate(theta)
        assert np.allclose(trig_value(product, theta), actual['product'], atol=1e-12, rtol=0)
        assert abs(trig_value(loss, theta) - actual['loss']) < 1e-12
        assert abs(trig_value(bw, theta) - actual['budget_w']) < 1e-12
        assert abs(trig_value(ba, theta) - actual['budget_a']) < 1e-12
    for lo, hi in zip(screened[:-1], screened[1:]):
        product, loss, bw, ba = analytic_cell_coefficients(p, (lo + hi) / 2)
        for frac in (.13, .43, .77):
            theta = lo + frac * (hi - lo)
            actual = p.evaluate(theta)
            assert np.allclose(trig_value(product, theta), actual['product'], atol=1e-12, rtol=0)
            assert abs(trig_value(loss, theta) - actual['loss']) < 1e-12


@pytest.mark.parametrize('group_size,pair', [(16,(3,11)),(8,(2,5)),(8,(2,11)),(4,(1,9))])
def test_screening_matches_full_partition_and_cell_algebra(group_size, pair):
    rng = np.random.default_rng(33 + group_size)
    W, X = rng.normal(size=(4,16)), rng.normal(size=(16,3))
    p = PairProblem(W, X, pair, group_size=group_size)
    full, fs = enumerate_events(p, False)
    screened, ss = enumerate_events(p, True)
    assert len(full) == len(screened)
    assert np.allclose(full, screened, atol=1e-10, rtol=0)
    assert ss['member_owner_interval_visits'] <= fs['member_owner_interval_visits']
    for theta in np.linspace(p.left,p.right,41):
        q = givens(16,pair,theta)
        assert np.linalg.norm(q.T@q-np.eye(16)) < 1e-12
        assert np.linalg.norm((W@q.T)@(q@X)-W@X)/np.linalg.norm(W@X) < 1e-12
        values = p.values(theta)
        for r in np.flatnonzero(p.stable):
            v0 = p.values(0)[r]
            assert abs(np.max(np.abs(values[r]))-np.max(np.abs(v0))) < 1e-12
    for left,right in zip(screened[:-1],screened[1:]):
        product,loss,bw,ba = analytic_cell_coefficients(p,(left+right)/2)
        for frac in (.23,.73):
            theta = left+frac*(right-left)
            actual = p.evaluate(theta)
            assert np.allclose(trig_value(product,theta),actual['product'],rtol=1e-10,atol=1e-11)
            assert abs(trig_value(loss,theta)-actual['loss']) < 1e-10
            assert abs(trig_value(bw,theta)-actual['budget_w']) < 1e-10
            assert abs(trig_value(ba,theta)-actual['budget_a']) < 1e-10


def test_solver_actual_quantization_budget_and_screening():
    rng = np.random.default_rng(20261115)
    p = PairProblem(rng.normal(size=(5,8)),rng.normal(size=(8,4)),(1,6))
    full = solve_pair_reference(p,False,True)
    screen = solve_pair_reference(p,True,True)
    assert abs(full['best']['loss']-screen['best']['loss']) < 1e-10
    for result in (full,screen):
        before,after = result['baseline'],result['best']
        assert after['loss'] <= before['loss'] + 1e-12
        assert after['budget_w'] <= before['budget_w'] + 1e-9
        assert after['budget_a'] <= before['budget_a'] + 1e-9
        q = givens(8,(1,6),after['theta'])
        a = quantize_absmax(p.W@q.T)[0]
        b = quantize_absmax((q@p.X).T)[0].T
        independently = np.sum((a@b-p.target)**2)/p.denominator
        assert abs(independently-after['loss']) < 1e-12
    # Dense grid is an independent comparator, not the solver or a certificate.
    base = p.evaluate(0.)
    grid = [p.evaluate(t) for t in np.linspace(p.left,p.right,1001)]
    feasible = [v for v in grid if v['budget_w']<=base['budget_w']+1e-10 and v['budget_a']<=base['budget_a']+1e-10]
    assert screen['best']['loss'] <= min(v['loss'] for v in feasible) + 1e-8


def test_report_has_real_evidence_and_explicit_limits(tmp_path):
    report = run_reference_checks(tmp_path)
    assert report['status'] == 'PASS'
    assert (tmp_path/'math_reference_report.json').is_file()
    assert not report['llm_experiment']
    assert not report['solution']['interval_certified']
    assert any('NOT_IMPLEMENTED' in item for item in report['limits'])
