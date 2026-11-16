"""Execution telemetry fixtures; never evidence of a pretrained-model run."""
import json
from pathlib import Path
import time

import numpy as np
import pytest

from cellquant.progress import RunProgress
from cellquant.runtime import read_json, atomic_json


def test_heartbeat_keeps_last_real_progress_and_terminal_state(tmp_path):
    progress = RunProgress(tmp_path, interval=.03)
    try:
        progress.step('g1.solve.pair', seed=11, method='cellquant', pair_index=3)
        before = read_json(tmp_path/'state.json')
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            after = read_json(tmp_path/'state.json')
            if after['event'] == 'heartbeat':
                break
            time.sleep(.01)
        assert after['event'] == 'heartbeat'
        assert after['progress_sequence'] == before['progress_sequence']
        assert after['last_progress_at_utc'] == before['last_progress_at_utc']
        assert after['phase'] == before['phase']
        assert after['process_rss_gib'] > 0
        assert after['process_cpu_seconds'] >= before['process_cpu_seconds']
    finally:
        progress.close('RESOURCE_BLOCKED')
    final = read_json(tmp_path/'state.json')
    assert final['status'] == 'RESOURCE_BLOCKED'
    assert not progress.thread.is_alive()
    with pytest.raises(RuntimeError, match='closed run'):
        progress.step('not permitted')


def test_heartbeat_io_failure_is_surfaced(tmp_path, monkeypatch):
    import cellquant.progress as module
    progress = RunProgress(tmp_path, interval=.01)
    def broken(*args):
        raise OSError('fixture disk failure')
    monkeypatch.setattr(module, 'atomic_json', broken)
    progress.thread.join(timeout=3)
    assert not progress.thread.is_alive()
    with pytest.raises(RuntimeError, match='heartbeat failed'):
        progress.close('FAILED')


@pytest.mark.parametrize('method', ['hadamard_shared', 'cellquant', 'angle_grid_matched',
                                   'givens_ste_matched', 'fixed_scale_search_matched'])
def test_solver_reports_real_work_without_changing_result(method):
    pytest.importorskip('torch')
    from cellquant.solver import CalibrationPanel, SolverConfig
    from cellquant.quantization import QuantSpec
    from cellquant.baselines import fit_baseline
    rng = np.random.default_rng(20261116)
    panels = [CalibrationPanel('input', 'input', rng.normal(size=(2,4)), rng.normal(size=(4,3))),
              CalibrationPanel('output', 'output', rng.normal(size=(4,8)), rng.normal(size=(8,3)))]
    options = dict(config=SolverConfig(theta_limit=.1, wall_seconds=120), grid_points=5, ste_steps=2)
    args = (method, panels, np.eye(4), QuantSpec(group_size=4), [(0, 1)])
    original = fit_baseline(*args, **options)
    events = []
    observed = fit_baseline(*args, **options, progress=lambda phase, **d: events.append((phase, d)))
    np.testing.assert_array_equal(original['q'], observed['q'])
    assert observed['final'] == original['final']
    assert observed['status'] == original['status']
    assert [d['panel_id'] for p, d in events if p == 'solve.initialize_panel'] == ['input','output']
    if method != 'hadamard_shared':
        assert any(p == 'solve.pair' and d['pair_index'] == 0 for p, d in events)


def test_legacy_and_stale_states_are_not_claimed_live(tmp_path):
    from cellquant.diagnostics import inspect_runs
    run = tmp_path/'runs/experiments/fixture'
    atomic_json(run/'state.json', {'status': 'RUNNING', 'phase': 'fusion'})
    before = (run/'state.json').read_bytes()
    report = inspect_runs(tmp_path, sample_seconds=.01)
    assert report['processes'] == []
    assert not report['recent_runs'][0]['live_process_verified']
    assert 'Legacy' in report['recent_runs'][0]['note']
    assert (run/'state.json').read_bytes() == before


def test_capture_progress_preserves_complete_output_axis():
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    from cellquant.capture import capture_panels
    from cellquant.data import token_hash
    class Wrapper(torch.nn.Module):
        def __init__(self, a, b):
            super().__init__();self.module = torch.nn.Linear(a, b, bias=False)
        def forward(self, x):return self.module(x)
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module();self.self_attn.q_proj = Wrapper(4, 4)
            self.mlp = torch.nn.Module();self.mlp.up_proj = Wrapper(4, 8);self.mlp.down_proj = Wrapper(8, 4)
        def forward(self, x):
            return (self.mlp.down_proj(self.mlp.up_proj(self.self_attn.q_proj(x))),)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(8, 4);self.model.norm = torch.nn.Identity()
            self.model.layers = torch.nn.ModuleList([Layer(), Layer()])
            self.config = SimpleNamespace(use_cache=False)
            self._cellquant_online_installed = True  # Control-flow fixture, not a SpinQuant equivalence claim.
        def forward(self, ids, **kwargs):
            x = self.model.embed_tokens(ids)
            for layer in self.model.layers:x = layer(x)[0]
            return x
    tokens = np.arange(4)
    blocks = [dict(tokens=tokens, block_id='fixture:0', document_id='fixture', token_ids_sha256=token_hash(tokens))]
    model = Model();events = []
    options = dict(layers=[0, 1], tokens_per_panel=2, input_weight_rows=2, device='cpu',
                   graph_identity='unit_test_fixture', source_type='unit_test_fixture')
    before = capture_panels(model, blocks, **options)
    after = capture_panels(model, blocks, **options, progress=lambda phase, **d:events.append((phase, d)))
    assert [d['layer'] for p, d in events if p == 'capture.layer'] == [0, 1]
    assert events[-1] == ('capture.complete', {'panels':6})
    for old, new in zip(before, after):
        np.testing.assert_array_equal(old.weight, new.weight)
        np.testing.assert_array_equal(old.activation, new.activation)
        if new.placement == 'output':assert new.weight.shape == (4, 8)


def test_diagnostics_checks_process_identity_and_does_not_expose_argv(tmp_path, monkeypatch):
    import cellquant.diagnostics as module
    import psutil
    from types import SimpleNamespace
    class Process:
        pid = 12345
        def cmdline(self):return ['python', '-u', '-m', 'cellquant.experiment', '--fixture-secret']
        def cwd(self):return str(tmp_path)
        def cpu_percent(self, interval):return 75.
        def create_time(self):return 100.
        def status(self):return 'running'
        def cpu_times(self):return SimpleNamespace(user=1., system=2.)
        def memory_info(self):return SimpleNamespace(rss=2**30)
    monkeypatch.setattr(psutil, 'process_iter', lambda: [Process()])
    run = tmp_path/'runs/experiments/fixture'
    atomic_json(run/'state.json', {'status':'RUNNING', 'pid':12345, 'process_create_time':100.,
                                  'progress_sequence':1, 'phase':'g1.solve.pair'})
    report = module.inspect_runs(tmp_path, sample_seconds=.001)
    assert report['processes'][0]['cpu_percent'] == 75.
    assert report['recent_runs'][0]['live_process_verified']
    assert '--fixture-secret' not in json.dumps(report)
    state = read_json(run/'state.json');state['process_create_time'] = 99.
    atomic_json(run/'state.json', state)
    assert not module.inspect_runs(tmp_path, sample_seconds=.001)['recent_runs'][0]['live_process_verified']
