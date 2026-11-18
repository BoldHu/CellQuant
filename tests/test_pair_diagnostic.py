import numpy as np
import pytest

torch = pytest.importorskip('torch')
from cellquant import experiment, pair_diagnostic as diagnostic
from cellquant.events import ResourceLimit
from cellquant.quantization import QuantSpec
from cellquant.runtime import read_json
from cellquant.solver import CalibrationPanel, SolverConfig
from cellquant.tensor_solver import PanelState, aggregate, hadamard_rotation, rotate_rows


def panels():
    rng = np.random.default_rng(45)
    return [CalibrationPanel(name, side, rng.normal(size=w), rng.normal(size=x),
                             objective_weight=weight, metadata={'source_type': 'unit_test_fixture'})
            for name, side, w, x, weight in [('q', 'input', (3, 4), (4, 3), .3),
                                              ('down', 'output', (4, 8), (8, 3), 1.4)]]


def test_output_support_is_an_upper_bound_and_relabels_the_same_vertices():
    q0 = torch.eye(4, dtype=torch.float64)
    spec = QuantSpec(group_size=4)
    states = [PanelState(p, q0, spec, spec) for p in panels()]
    schedule = [(0, 1), (1, 2), (1, 0)]
    ranked, summary = diagnostic.compare_support(states, schedule)
    assert diagnostic.compare_support(states[1:], schedule)[0] == ranked
    assert ranked[0][1] == ranked[1][0] and ranked[0][::-1] == ranked[2]
    assert len({axis for pair in ranked for axis in pair}) == summary['distinct_axes'] == 3
    output = states[1]
    initial = float(output.loss)
    original_error = output.error.clone()
    coverage = summary['output_objective_bound_fraction']
    assert 0 <= coverage['original'] <= coverage['ranked'] <= 1 + 1e-12
    for pair, theta in zip(ranked, [.4, -.2, .7]):
        output.accept(pair, theta)
    untouched = list(set(range(4)) - set(summary['ranked_axes']))
    torch.testing.assert_close(output.error[untouched], original_error[untouched], rtol=0, atol=0)
    assert (initial - float(output.loss)) / initial <= coverage['ranked'] + 1e-12


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_each_probe_compares_full_grid_and_events_at_the_same_initial_rotation(device, tmp_path):
    items, spec = panels(), QuantSpec(group_size=4)
    q0 = hadamard_rotation(4, 11, device)
    q_before = q0.clone()
    cfg = SolverConfig(backend='torch', device=device, theta_limit=.12, seconds_per_pair=45, wall_seconds=180)
    path = tmp_path / 'seed_11' / 'pair_diagnostic.json'
    result = diagnostic.diagnose_seed(items, spec, q0, [(0, 1), (1, 2)], cfg, seed=11, output_path=path)
    assert result == read_json(path) and result['status'] == 'COMPLETE'
    assert result['source_type'] == 'unit_test_fixture' and not list(tmp_path.rglob('G1*.json'))
    assert len(result['probes']) == 8 and result['probe_positions_zero_based'] == [0, 1]
    torch.testing.assert_close(q0, q_before, rtol=0, atol=0)
    for row in result['probes']:
        q = q0.clone()
        rotate_rows(q, row['pair'], row['best']['theta'])
        expected = aggregate([PanelState(p, q, spec, spec) for p in items])
        stepped = [PanelState(p, q0, spec, spec) for p in items]
        for state in stepped:
            state.accept(row['pair'], row['best']['theta'])
        incremental = aggregate(stepped)
        for key, value in expected.items():
            assert row['replayed'][key] == pytest.approx(float(value), rel=1e-10, abs=1e-12)
            assert row['best'][key] == pytest.approx(float(incremental[key]), rel=1e-10, abs=1e-12)
            assert row['best'][key] <= result['initial'][key] + 1e-10
        assert sum(row['replayed_joint_loss_drop_by_side'].values()) == pytest.approx(
            result['initial']['loss'] - row['replayed']['loss'], abs=1e-12)
        assert row['feasible_replay_gain_pct'] >= 0
        if row['method'] == 'grid':
            assert row['grid_evaluations'] == 129
        else:
            assert row['event_candidates'] > 0


def test_incomplete_pair_preserves_the_previous_measured_probe(tmp_path, monkeypatch):
    def exhausted(*args, **kwargs):
        raise ResourceLimit('fixture event budget')
    monkeypatch.setattr(diagnostic, 'solve_pair', exhausted)
    path = tmp_path / 'partial.json'
    with pytest.raises(ResourceLimit, match='fixture event budget'):
        diagnostic.diagnose_seed(panels(), QuantSpec(), torch.eye(4, dtype=torch.float64), [(0, 1)],
                                 SolverConfig(theta_limit=.12), seed=11, output_path=path)
    partial = read_json(path)
    assert partial['status'] == 'RESOURCE_BLOCKED' and len(partial['probes']) == 1
    assert partial['last_probe'] == {'position': 0, 'arm': 'original', 'method': 'cellquant'}
    assert 'mean_single_pair_gain_pct' not in partial


def test_fit_diagnostic_cannot_start_another_stage_or_write_the_old_result(tmp_path, monkeypatch):
    cfg = {'stage': 'full', 'result_path': 'formal_result.json', 'solver': {}}
    old_result = tmp_path / cfg['result_path']
    old_result.write_text('preserved')
    def preflight(config, root):
        assert 'result_path' not in config
        return {'status': 'READY', 'missing': []}
    monkeypatch.setattr(experiment, 'preflight', preflight)
    result = experiment.run_experiment(cfg, project_root=tmp_path, diagnose_pairs=True)
    assert result['status'] == 'BLOCKED' and old_result.read_text() == 'preserved'
    with pytest.raises(ValueError, match='either'):
        experiment.run_experiment(cfg, project_root=tmp_path, diagnose_pairs=True, profile_pair=True)
