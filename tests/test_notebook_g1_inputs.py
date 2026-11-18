"""Notebook execution guards using local fixtures, never HF/GPU/gate evidence."""
import ast
import json
import os
import sys
from pathlib import Path

import pytest

from cellquant import experiment

ROOT = Path(__file__).resolve().parents[1]


def cell(name, marker):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return next("".join(c["source"]) for c in notebook["cells"]
                if c["cell_type"] == "code" and marker in "".join(c["source"]))


def pilot_config_cell(run_real):
    tree = ast.parse(cell("01_Colab_Real_Layer_Pilot.ipynb", "preflight = run_experiment("))
    # Test both UI switch values without changing the user's saved choice.
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RUN_REAL_MODEL"
                                               for t in node.targets):
            node.value = ast.Constant(run_real)
    return compile(ast.fix_missing_locations(tree), "pilot-config-cell", "exec")


def context(tmp_path):
    return {"PROJECT_ROOT": tmp_path, "PATHS": {"runs": tmp_path / "runs"},
            "Path": Path, "os": os, "sys": sys}


def test_workspace_defaults_and_authentication_do_not_persist_secrets(tmp_path,monkeypatch):
    import re
    name="00_Colab_GitHub_Workspace.ipynb"
    assert not re.search(r"hf_[A-Za-z0-9]{24,}",(ROOT/"notebooks"/name).read_text(encoding="utf-8"))
    monkeypatch.setenv("HF_TOKEN","unit-test-secret")
    ns=context(tmp_path)
    exec(cell(name,"REPO_URL ="),ns)
    assert ns["HF_TOKEN"]=="unit-test-secret" and ns["PREPARE_G1_INPUTS"] is False
    # With preparation off, even an existing env credential causes no Hub call.
    monkeypatch.setitem(sys.modules,"huggingface_hub",None)
    exec(cell(name,"from huggingface_hub import HfApi, login"),ns)


@pytest.mark.parametrize("run_real", [False, True])
def test_missing_prepared_config_never_starts_a_real_run(tmp_path, monkeypatch, run_real):
    requested = tmp_path / "runs/prepared/g1/config.json"
    monkeypatch.setenv("CELLQUANT_PILOT_CONFIG", str(requested))
    calls = []

    def preflight(path, **kw):
        calls.append((path, kw))
        return {"readiness": {"status": "BLOCKED", "missing": ["fixture: inputs missing"]}}

    monkeypatch.setattr(experiment, "run_experiment", preflight)
    ns = context(tmp_path)
    if run_real:
        with pytest.raises(FileNotFoundError, match="PREPARE_G1_INPUTS"):
            exec(pilot_config_cell(True), ns)
        assert not calls
        assert ns["PILOT_CONFIG_PATH"] == requested
    else:
        exec(pilot_config_cell(False), ns)
        assert calls == [(tmp_path / "configs/colab_pilot.json",
                          {"project_root": tmp_path, "preflight_only": True})]
        assert ns["preflight"]["readiness"]["status"] == "BLOCKED"


def test_prepared_config_is_used_and_missing_reasons_block_execution(tmp_path, monkeypatch):
    cfg = tmp_path / "runs/prepared/custom/config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({'source_type':'unit_test_fixture',
        'runtime':{'cpu_offload':False,'model_device':'cuda:0'},
        'solver':{'backend':'torch','device':'cuda:0'}}))
    monkeypatch.setenv("CELLQUANT_PILOT_CONFIG", str(cfg.relative_to(tmp_path)))
    calls = []

    def preflight(path, **kw):
        calls.append((path, kw))
        return {"readiness": {"status": "BLOCKED", "missing": ["fixture: insufficient CPU RAM"]}}

    monkeypatch.setattr(experiment, "run_experiment", preflight)
    ns = context(tmp_path)
    exec(pilot_config_cell(True), ns)
    assert calls == [(cfg, {"project_root": tmp_path, "preflight_only": True})]

    def forbidden(*args, **kwargs):
        raise AssertionError("A blocked pilot must not launch a subprocess")

    ns["stream_command"] = forbidden
    with pytest.raises(RuntimeError, match="fixture: insufficient CPU RAM") as error:
        exec(cell("01_Colab_Real_Layer_Pilot.ipynb", "process = stream_command("), ns)
    assert str(cfg) in str(error.value)


def test_old_cpu_configuration_is_rejected_before_real_launch(tmp_path,monkeypatch):
    cfg=tmp_path/'old_cpu_config.json'
    cfg.write_text(json.dumps({'runtime':{'cpu_offload':True},'solver':{}}))
    monkeypatch.setenv('CELLQUANT_PILOT_CONFIG',str(cfg))
    with pytest.raises(RuntimeError,match='g1_gpu'):
        exec(pilot_config_cell(True),context(tmp_path))


@pytest.mark.parametrize("enabled,mounted", [(False, True), (True, False)])
def test_preparation_requires_execution_switch_and_workspace(tmp_path, monkeypatch, enabled, mounted):
    monkeypatch.delenv("CELLQUANT_PILOT_CONFIG", raising=False)
    ns = context(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation must not run")

    ns.update(PREPARE_G1_INPUTS=enabled, WORKSPACE_READY=mounted, logged_command=forbidden)
    script = cell("00_Colab_GitHub_Workspace.ipynb", "if PREPARE_G1_INPUTS:")
    if enabled:
        with pytest.raises(RuntimeError, match="Drive"):
            exec(script, ns)
    else:
        exec(script, ns)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("reuse", [False, True])
def test_prepare_verify_backup_order_and_frozen_config_preservation(tmp_path, monkeypatch, reuse):
    cfg = tmp_path / "runs/prepared/g1/config.json"
    frozen = '{"source_type":"unit_test_fixture","immutable_test_marker":12}'
    if reuse:
        cfg.parent.mkdir(parents=True)
        cfg.write_text(frozen)
    monkeypatch.setenv("CELLQUANT_PILOT_CONFIG", str(cfg))
    calls = []

    def command(argv, label):
        calls.append(label)
        if label == "prepare_g1":
            assert not cfg.exists()
            assert argv[argv.index("--output") + 1] == str(cfg.parent)
            cfg.parent.mkdir(parents=True)
            cfg.write_text(frozen)
        elif label == "cache_g1_model":
            assert argv[argv.index("--config") + 1] == str(cfg)
        elif label == "backup_g1_inputs":
            assert argv[argv.index("--areas") + 1:] == ["models", "datasets", "runs"]

    def preflight(path, **kw):
        assert path == cfg and kw["preflight_only"] is True
        calls.append("preflight")
        return {"readiness": {"status": "READY", "missing": []}}

    monkeypatch.setattr(experiment, "run_experiment", preflight)
    ns = context(tmp_path)
    ns.update(PREPARE_G1_INPUTS=True, WORKSPACE_READY=True, logged_command=command)
    exec(cell("00_Colab_GitHub_Workspace.ipynb", "if PREPARE_G1_INPUTS:"), ns)
    assert calls == ([] if reuse else ["prepare_g1"]) + ["preflight", "cache_g1_model", "backup_g1_inputs"]
    assert cfg.read_text() == frozen


@pytest.mark.parametrize("failure_phase", ["prepare_g1", "preflight", "cache_g1_model"])
def test_preparation_or_weight_failure_stops_before_backup(tmp_path, monkeypatch, failure_phase, capsys):
    monkeypatch.delenv("CELLQUANT_PILOT_CONFIG", raising=False)
    calls = []

    def command(argv, label):
        calls.append(label)
        if label == failure_phase:
            raise RuntimeError("fixture failure")

    monkeypatch.setattr(experiment, "run_experiment", lambda *a, **k:
                        {"readiness": {"status": "BLOCKED" if failure_phase == "preflight" else "READY",
                                       "missing": ["fixture failure"] if failure_phase == "preflight" else []}})
    ns = context(tmp_path)
    ns.update(PREPARE_G1_INPUTS=True, WORKSPACE_READY=True, logged_command=command)
    with pytest.raises(RuntimeError, match="fixture failure"):
        exec(cell("00_Colab_GitHub_Workspace.ipynb", "if PREPARE_G1_INPUTS:"), ns)
    assert "backup_g1_inputs" not in calls
    assert "G1 输入准备、权重校验和 Drive 备份已完成" not in capsys.readouterr().out


@pytest.mark.parametrize('profile,legacy,returncode',[(False,False,2),(False,True,2),(True,False,2),(True,False,0)])
def test_notebook_reports_saved_failure_and_profile_never_scores_g1(tmp_path,profile,legacy,returncode,capsys):
    from cellquant.runtime import atomic_json,read_json
    ns=context(tmp_path);out=tmp_path/'logs';out.mkdir()
    run=tmp_path/'run';run.mkdir()
    reason='seed=11 cellquant: Pair wall-clock budget exceeded: 300s'
    atomic_json(run/'result.json',{'source_type':'unit_test_fixture',
        'status':'RESOURCE_BLOCKED' if returncode else 'PROFILE_COMPLETE','reason':None if legacy else reason})
    if legacy:
        atomic_json(run/'seed_11/cellquant/pilot_result.json',{'source_type':'unit_test_fixture',
            'seed':11,'method':'cellquant','status':'RESOURCE_BLOCKED',
            'solver':{'reason':'Pair wall-clock budget exceeded: 300s','log':[]}})
    commands=[]
    def command(argv,*args,**kwargs):
        commands.append(argv)
        atomic_json(argv[argv.index('--result-index')+1],{'run_dir':str(run)})
        return {'returncode':returncode}
    ns.update(OUT=out,RUN_REAL_MODEL=True,RUN_PAIR_PROFILE=profile,PROFILE_PAIR_COUNT=8,PILOT_CONFIG_PATH=tmp_path/'config.json',
        PILOT_RUN_DIR=None,preflight={'readiness':{'status':'READY','missing':[]}},
        stream_command=command,read_json=read_json,atomic_json=atomic_json)
    script=cell('01_Colab_Real_Layer_Pilot.ipynb','process = stream_command(')
    if returncode:
        with pytest.raises(RuntimeError,match='Pair wall-clock budget exceeded: 300s') as error:exec(script,ns)
        assert str(run) in str(error.value)
    else:exec(script,ns)
    assert ('--profile-pair' in commands[0])==profile
    if profile:assert commands[0][commands[0].index('--profile-pairs')+1]=='8'
    assert ns['PILOT_RUN_DIR']==run
    if profile:
        def forbidden(*args,**kwargs):raise AssertionError('Pair profile must not score G1')
        ns['evaluate_file']=forbidden
        exec(cell('01_Colab_Real_Layer_Pilot.ipynb','default_evidence = PILOT_RUN_DIR'),ns)
        assert 'NOT_RUN' in capsys.readouterr().out and not (out/'G1.json').exists()


def test_notebook_summary_uses_current_profile_run_and_reads_only(tmp_path,capsys):
    from cellquant.runtime import atomic_json
    run=tmp_path/'current_profile'
    atomic_json(run/'config.json',{'seeds':[11,23],'solver':{'pair_budget':128}})
    atomic_json(run/'pair_profile.json',{'source_type':'unit_test_fixture','status':'PROFILE_COMPLETE'})
    atomic_json(run/'pair_profile/seed_11/cellquant/profile_result.json',{
        'source_type':'unit_test_fixture','method':'cellquant','status':'COMPLETE',
        'solver':{'seconds':170,'requested_pairs':8,'completed_pairs':8,'log':[{'seconds':20}]*8}})
    ns=context(tmp_path);ns['PILOT_RUN_DIR']=run
    before={p:p.read_bytes() for p in run.rglob('*') if p.is_file()}
    exec(cell('01_Colab_Real_Layer_Pilot.ipynb','budget_report = analyze_profile_budget('),ns)
    output=capsys.readouterr().out
    assert str(run) in output and 'PROJECTED_SEARCH_EXCEEDS_METHOD_BUDGET' in output
    assert '2026-11-17T081006' not in output
    assert before=={p:p.read_bytes() for p in run.rglob('*') if p.is_file()}
