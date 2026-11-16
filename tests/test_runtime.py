from __future__ import annotations
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import pytest

from cellquant.runtime import (atomic_json, build_manifest, verify_manifest,
                               extract_project_archive, stream_command, copy_completed_run)
from cellquant.panels import (normalized_hadamard_last_axis, analyze_input_panel,
                              stable_group_statistics, dequantize_last_axis)
from cellquant.runner import make_job_manifest, check_execution_readiness, load_real_result_rows, execution_fingerprint


def test_manifest_detects_changed_artifact(tmp_path):
    source = tmp_path / "result.json"
    atomic_json(source, {"status": "measured", "value": 1})
    manifest = build_manifest(tmp_path, [source], config={"seed": 7})
    assert verify_manifest(tmp_path, manifest)["status"] == "PASS"
    atomic_json(source, {"status": "measured", "value": 2})
    assert verify_manifest(tmp_path, manifest)["status"] == "FAIL"


def test_missing_and_empty_manifests_cannot_pass(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path, ["missing.json"])
    manifest = build_manifest(tmp_path, [])
    assert verify_manifest(tmp_path, manifest)["status"] == "INCOMPLETE"


def test_copy_keeps_rotation_checkpoint_and_reports_skips(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.mkdir()
    (source / "R.bin").write_bytes(b"small rotation checkpoint")
    (source / "pytorch_model.bin").write_bytes(b"must not silently transfer base model")
    result = copy_completed_run(source, dest)
    assert (dest / "R.bin").read_bytes() == (source / "R.bin").read_bytes()
    assert any(r["path"] == "pytorch_model.bin" for r in result["skipped"])
    assert verify_manifest(dest, json.loads(Path(result["manifest"]).read_text()))["status"] == "PASS"


def test_resume_fingerprint_changes_with_code(tmp_path):
    (tmp_path / "src/cellquant").mkdir(parents=True)
    source = tmp_path / "src/cellquant/backend.py"
    source.write_text("VERSION = 1\n")
    first = execution_fingerprint(tmp_path)
    source.write_text("VERSION = 2\n")
    second = execution_fingerprint(tmp_path)
    assert first["sha256"] != second["sha256"]
    assert first["status"] == "INCOMPLETE"  # model/data provenance still absent


def test_archive_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.txt", "unsafe")
    with pytest.raises(ValueError):
        extract_project_archive(archive, tmp_path / "extract")


def test_timeout_terminates_without_stdout_deadlock(tmp_path):
    with pytest.raises(TimeoutError):
        stream_command([sys.executable, "-c", "import time; time.sleep(10)"],
                       tmp_path / "silent.log", timeout_s=.15, poll_interval_s=.02, echo=False)


def test_stream_command_keeps_output_and_nonzero_exit(tmp_path):
    log = tmp_path / "run.log"
    result = stream_command([sys.executable, "-u", "-c", "print('logged'); raise SystemExit(3)"],
                            log, echo=False)
    assert result["returncode"] == 3
    assert "logged" in log.read_text()


def test_full_feature_axis_and_input_hadamard_identity():
    rng = np.random.default_rng(8)
    W, X = rng.normal(size=(5, 32)), rng.normal(size=(7, 32))
    r = analyze_input_panel(W, X, group_size=8, pairs=4)
    assert r["full_precision_identity_relative_error"] < 1e-12
    assert r["weight_shape"] == [5, 32]
    assert r["activation_rows_shape"] == [7, 32]
    with pytest.raises(ValueError):
        normalized_hadamard_last_axis(np.zeros((2, 30)))
    with pytest.raises(ValueError):
        dequantize_last_axis(np.zeros((2, 30)), group_size=8)


def test_absmax_quantizer_zero_and_range():
    z = np.array([[0., 0., 0., 0.], [-7., 7., 3.49, 3.51]])
    q = dequantize_last_axis(z)
    assert np.all(np.isfinite(q))
    assert np.array_equal(q[0], np.zeros(4))
    assert np.array_equal(q[1], [-7, 7, 3, 4])


def test_missing_backend_or_gate_is_not_ready(tmp_path):
    cfg = tmp_path / "cfg.json"
    atomic_json(cfg, {"stage": "full"})
    plan = make_job_manifest([{"job_id": "x", "config_path": str(cfg)}], project_root=tmp_path,
                             backend_module="cellquant.nonexistent_backend")
    ready = check_execution_readiness(plan, gate_path=tmp_path / "missing.json")
    assert ready["status"] == "BLOCKED"


def test_synthetic_or_paper_results_are_rejected(tmp_path):
    p = tmp_path / "fake.json"
    atomic_json(p, {"status": "COMPLETE", "evidence_kind": "SYNTHETIC", "metrics": {"ppl": 5}})
    result = load_real_result_rows([p])
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert not result["rows"]


def test_old_gpu_g3_cannot_unlock_new_gpu_manifest(tmp_path, monkeypatch):
    import cellquant.runner as runner
    import cellquant.gates as gates
    from cellquant.runtime import sha256_file
    # Isolated control-flow fixture: not exported and never usable as research evidence.
    monkeypatch.setattr(runner, "backend_available", lambda module: True)
    monkeypatch.setattr(runner, "execution_fingerprint", lambda *args, **kwargs:
                        {"status": "VERIFIED", "missing": [], "sha256": "new_gpu", "payload": {}})
    cfg = atomic_json(tmp_path / "cfg.json", {"fixture": True})
    plan = runner.make_job_manifest([{"job_id": "fixture", "config_path": str(cfg)}],
                                    project_root=tmp_path, required_gate="G3")
    evidence = atomic_json(tmp_path / "evidence.json", {
        "provenance": {"execution_fingerprint_sha256": "old_gpu", "unit_test_fixture": True}})
    gate = atomic_json(tmp_path / "gate.json", {
        "gate_id": "G3", "status": "PASS", "rules_sha256": "rules_fixture",
        "evidence_sha256": sha256_file(evidence)})
    monkeypatch.setattr(gates, "evaluate_file", lambda *args, **kwargs:
                        {"status": "PASS", "rules_sha256": "rules_fixture"})
    report = runner.check_execution_readiness(plan, gate_path=gate, gate_evidence_path=evidence)
    assert report["status"] == "BLOCKED"
    assert any("G3 is not bound" in message for message in report["missing"])


def test_changed_gate_rules_block_old_report(tmp_path, monkeypatch):
    import cellquant.runner as runner
    import cellquant.gates as gates
    from cellquant.runtime import sha256_file
    monkeypatch.setattr(runner, "backend_available", lambda module: True)
    monkeypatch.setattr(runner, "execution_fingerprint", lambda *args, **kwargs:
                        {"status": "VERIFIED", "missing": [], "sha256": "current", "payload": {}})
    cfg = atomic_json(tmp_path / "cfg.json", {"fixture": True})
    plan = runner.make_job_manifest([{"job_id": "fixture", "config_path": str(cfg)}],
                                    project_root=tmp_path, required_gate="G1")
    evidence = atomic_json(tmp_path / "evidence.json", {"unit_test_fixture": True})
    gate = atomic_json(tmp_path / "gate.json", {
        "gate_id": "G1", "status": "PASS", "rules_sha256": "old_rules",
        "evidence_sha256": sha256_file(evidence)})
    monkeypatch.setattr(gates, "evaluate_file", lambda *args, **kwargs:
                        {"status": "PASS", "rules_sha256": "new_rules"})
    report = runner.check_execution_readiness(plan, gate_path=gate, gate_evidence_path=evidence)
    assert report["status"] == "BLOCKED"
    assert any("rules changed" in message for message in report["missing"])
