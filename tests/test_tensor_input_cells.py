"""Compare the dynamic input batch path to independent direct RTN operations."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from cellquant import tensor_solver as ts
from cellquant import tensor_input_cells as batch
from cellquant.tensor_cells import input_term
from cellquant.tensor_math import trig_value
from cellquant.quantization import QuantSpec
from cellquant.solver import CalibrationPanel, SolverConfig
from cellquant.events import WorkBudget


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


def make_state(device, case, width=16):
    rng = np.random.default_rng(1701)
    w = rng.normal(size=(8, width))*.15
    x = rng.normal(size=(width, 9))*.15
    w[:, 2] = 4
    x[2] = 4
    if case in ('weight', 'bilateral', 'ties_zero'):
        w[0, 2] = .3
        w[0, [0, 5]] = [5, 3]
    if case in ('activation', 'bilateral', 'ties_zero'):
        x[2, 0] = .3
        x[[0, 5], 0] = [5, 3]
    if case == 'ties_zero':
        w[0, 3:5] = [2.5, -2.5]
        x[3:5, 0] = [2.5, -2.5]
        w[1] = 0
        x[:, 1] = 0
    p = CalibrationPanel('full.input', 'input', w, x)
    spec = QuantSpec(group_size=-1)
    return ts.PanelState(p, torch.eye(width, dtype=torch.float64, device=device), spec, spec)


@pytest.mark.parametrize('case', ['weight', 'activation', 'bilateral', 'ties_zero'])
@pytest.mark.parametrize('chunk', [1, 37])
def test_dynamic_batch_keeps_codes_cross_terms_boundaries_and_risks(device, case, chunk):
    s = make_state(device, case)
    orbit = ts.PanelOrbit(s, (0, 5))
    assert batch.can_batch(orbit)
    work = WorkBudget(200000, 120)
    events = [g.enumerate(-.78, .78, True, work)[:2] for g in orbit.groups]
    term = input_term(orbit, events, work, lambda *a, **k: None, chunk)
    assert orbit.update_stats['input_batched_full_row_cells'] == len(term.loss)
    assert orbit.update_stats['full_gemm_per_event'] == 0
    # Every boundary, including simultaneous events, plus interior points
    # independent of the midpoints used to identify the frozen code branch.
    for i, theta in enumerate(term.angles):
        actual = s.direct((0, 5), theta)
        expected = torch.stack([actual[k] for k in ('loss', 'budget_w', 'budget_a')])
        torch.testing.assert_close(term.boundary[i], expected, atol=2e-11, rtol=2e-10)
        if i < len(term.loss):
            theta = theta+(term.angles[i+1]-theta)*.37
            actual = s.direct((0, 5), theta)
            expected = torch.stack([actual[k] for k in ('loss', 'budget_w', 'budget_a')])
            predicted = torch.stack([trig_value(v[i], theta) for v in (term.loss, term.bw, term.ba)])
            torch.testing.assert_close(predicted, expected, atol=2e-11, rtol=2e-10)


def test_batched_and_original_incremental_input_coefficients_match(device, monkeypatch):
    s = make_state(device, 'bilateral')
    work = WorkBudget(200000, 120)
    orbit = ts.PanelOrbit(s, (0, 5))
    events = [g.enumerate(-.78, .78, True, work)[:2] for g in orbit.groups]
    actual = input_term(orbit, events, work, lambda *a, **k: None, 31)
    monkeypatch.setattr(batch, 'can_batch', lambda orbit: False)
    reference = input_term(ts.PanelOrbit(s, (0, 5)), events, work, lambda *a, **k: None, 31)
    for name in ('angles', 'loss', 'bw', 'ba', 'boundary'):
        torch.testing.assert_close(getattr(actual, name), getattr(reference, name), atol=2e-11, rtol=2e-10)


def test_grouped_and_dense_unstable_cases_use_existing_path(device):
    s = make_state(device, 'bilateral')
    grouped = ts.PanelState(s.panel, torch.eye(16, dtype=torch.float64, device=device),
                            QuantSpec(group_size=4), QuantSpec(group_size=4))
    assert not batch.can_batch(ts.PanelOrbit(grouped, (0, 5)))
    s.a[:, 0] = 100
    s.x[0] = 100
    s.refresh()
    assert not batch.can_batch(ts.PanelOrbit(s, (0, 5)))


@pytest.mark.parametrize('group_size,pair', [(-1, (5, 0)), (4, (1, 0)), (4, (5, 0))])
@pytest.mark.parametrize('fixed', [False, True])
def test_batched_stable_events_preserve_group_member_and_budget(device, group_size, pair, fixed):
    s = make_state(device, 'ties_zero')
    q = torch.eye(s.panel.hidden_size, dtype=torch.float64, device=device)
    spec = QuantSpec(group_size=group_size)
    state = ts.PanelState(s.panel, q, spec, spec, fixed=fixed)
    state.accept((2, 3), .21)
    orbit = ts.PanelOrbit(state, pair)
    batched_work, scalar_work = WorkBudget(), WorkBudget()
    events = batch.stable_group_events(orbit, -.78, .78, batched_work)
    assert events and set(events) == {i for i, g in enumerate(orbit.groups) if g.stable}
    for gid, (angles, members, stats) in events.items():
        ref_angles, ref_members, ref_stats = orbit.groups[gid].enumerate(-.78, .78, True, scalar_work)
        assert stats == ref_stats
        # Compare every incidence, including endpoints and duplicated roots.
        actual = sorted(zip(members.tolist(), angles.tolist()))
        expected = sorted(zip(ref_members.tolist(), ref_angles.tolist()))
        np.testing.assert_array_equal(np.array(actual)[:, 0], np.array(expected)[:, 0])
        np.testing.assert_allclose(np.array(actual)[:, 1], np.array(expected)[:, 1], atol=2e-15, rtol=0)
    assert batched_work.generated == scalar_work.generated


def test_full_hidden_axis_dynamic_batch(device):
    s = make_state(device, 'bilateral', 4096)
    orbit = ts.PanelOrbit(s, (0, 5))
    work = WorkBudget(200000, 120)
    events = [g.enumerate(-.04, .04, True, work)[:2] for g in orbit.groups]
    term = input_term(orbit, events, work, lambda *a, **k: None, 23)
    assert orbit.update_stats['input_batched_full_row_cells'] == len(term.loss)
    assert s.a.shape[1] == s.x.shape[0] == 4096
    for i in torch.linspace(0, len(term.angles)-1, min(20, len(term.angles)), device=device).long():
        theta = term.angles[i]
        actual = s.direct((0, 5), theta)
        torch.testing.assert_close(term.boundary[i], torch.stack([actual[k] for k in ('loss', 'budget_w', 'budget_a')]),
                                   atol=2e-11, rtol=2e-10)


def test_continuous_shared_rotation_matches_original_search(device, monkeypatch):
    s = make_state(device, 'bilateral')
    other = CalibrationPanel('second.input', 'input', s.panel.weight*.8,
                             s.panel.activation*1.1, objective_weight=.7)
    q = torch.eye(16, dtype=torch.float64, device=device)
    cfg = SolverConfig(backend='torch', device=device, theta_limit=.2,
                       seconds_per_pair=120, wall_seconds=240, cell_batch_size=31)
    schedule = [(0, 5), (0, 7)]
    panels = [s.panel, other]
    actual = ts.fit_baseline('cellquant', panels, q, QuantSpec(group_size=-1), schedule, cfg)
    assert any(st.get('input_batched_full_row_cells', 0) for row in actual['log']
               for st in row['incremental_statistics'])
    monkeypatch.setattr(batch, 'can_batch', lambda orbit: False)
    reference = ts.fit_baseline('cellquant', panels, q, QuantSpec(group_size=-1), schedule, cfg)
    assert actual['status'] == reference['status'] == 'COMPLETE'
    assert actual['completed_pairs'] == reference['completed_pairs'] == len(schedule)
    for key in ('loss', 'budget_w', 'budget_a'):
        assert actual['final'][key] == pytest.approx(reference['final'][key], abs=2e-10, rel=1e-8)
    torch.testing.assert_close(actual['q'], reference['q'], atol=2e-9, rtol=2e-9)
