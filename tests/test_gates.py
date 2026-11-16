"""Synthetic logic tests only; fixtures are never promotable experiment evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cellquant.gates import (
    BLOCKED, FAIL, INSUFFICIENT, PASS, evaluate_evidence, evaluate_file,
    evaluate_metrics, load_rules, main, validate_rules,
)


def cond(metric, op, value):
    return {"metric": metric, "op": op, "value": value,
            "unit": "unit_test_fixture", "rationale": "Synthetic threshold logic fixture, not a result."}


@pytest.fixture
def gate_fixture():
    return {
        "gate_id": "G1", "title": "unit_test_fixture", "on_pass": "unit_test_only",
        "allowed_source_types": ["real_layer", "measured_gpu"],
        "common_checks": [cond("correct", "eq", True), cond("error", "le", 0.1)],
        "routes": {
            "accuracy": [cond("gain", "ge", 5.0)],
            "efficiency": [cond("speedup", "ge", 2.0)],
        },
    }


@pytest.fixture
def rule_fixture(gate_fixture):
    return {"schema_version": "1.0", "gates": [gate_fixture]}


@pytest.fixture
def evidence_fixture():
    # Never saved as PASS and never retagged as genuine research measurements.
    return {
        "schema_version": "1.0", "gate_id": "G1", "source_type": "unit_test_fixture",
        "metrics": {"correct": True, "error": 0.01, "gain": 6.0, "speedup": 2.5},
        "provenance": {"kind": "unit_test_fixture", "stage": "G1"},
        "artifact_refs": ["unit_test_fixture:no_real_measurement"],
    }


def test_one_passing_route_does_not_require_other_route(gate_fixture):
    result = evaluate_metrics(gate_fixture, {"correct": True, "error": 0.01, "gain": 6})
    assert result["status"] == PASS
    assert result["routes"]["efficiency"]["status"] == INSUFFICIENT
    assert result["passing_routes"] == ["accuracy"]
    assert "metrics.speedup" in result["missing_fields"]


def test_route_failure_cannot_block_other_valid_route(gate_fixture):
    result = evaluate_metrics(gate_fixture, {"correct": True, "error": 0.01, "gain": 3, "speedup": 3})
    assert result["status"] == PASS
    assert result["passing_routes"] == ["efficiency"]


def test_common_failure_cannot_be_overridden(gate_fixture):
    result = evaluate_metrics(gate_fixture, {"correct": False, "error": 0.01, "gain": 10, "speedup": 10})
    assert result["status"] == FAIL


def test_all_routes_fail(gate_fixture):
    result = evaluate_metrics(gate_fixture, {"correct": True, "error": 0.01, "gain": 1, "speedup": 1})
    assert result["status"] == FAIL


def test_one_failed_and_one_unknown_route_is_insufficient(gate_fixture):
    result = evaluate_metrics(gate_fixture, {"correct": True, "error": 0.01, "gain": 1})
    assert result["status"] == INSUFFICIENT


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf"), "0", True, [], {}])
def test_bad_numeric_measurements_do_not_coerce_to_zero(gate_fixture, bad):
    result = evaluate_metrics(gate_fixture, {"correct": True, "error": bad, "gain": 10})
    assert result["status"] == INSUFFICIENT
    assert "metrics.error" in result["missing_fields"]
    json.dumps(result, allow_nan=False)  # Even error reports are strict JSON.


def test_boolean_one_is_not_true(gate_fixture):
    result = evaluate_metrics(gate_fixture, {"correct": 1, "error": 0.01, "gain": 10})
    assert result["status"] == INSUFFICIENT


def test_missing_error_not_default_zero(gate_fixture):
    result = evaluate_metrics(gate_fixture, {"correct": True, "gain": 10})
    assert result["status"] == INSUFFICIENT


def test_fixture_never_promoted(rule_fixture, evidence_fixture):
    result = evaluate_evidence("G1", evidence_fixture, rule_fixture)
    assert result["status"] == BLOCKED
    assert not result["passing_routes"]


def test_fixture_cannot_be_enabled_by_custom_source_rule(rule_fixture, evidence_fixture):
    rule_fixture["gates"][0]["allowed_source_types"].append("unit_test_fixture")
    assert evaluate_evidence("G1", evidence_fixture, rule_fixture)["status"] == BLOCKED


def test_fixture_provenance_prevents_relabeling(rule_fixture, evidence_fixture):
    evidence_fixture["source_type"] = "real_layer"
    assert evaluate_evidence("G1", evidence_fixture, rule_fixture)["status"] == BLOCKED


@pytest.mark.parametrize("source", ["published", "paper_reported", "estimate", "mock"])
def test_external_or_mock_results_blocked(rule_fixture, evidence_fixture, source):
    evidence_fixture["source_type"] = source
    assert evaluate_evidence("G1", evidence_fixture, rule_fixture)["status"] == BLOCKED


def test_wrong_gate_id_blocked(rule_fixture, evidence_fixture):
    evidence_fixture["gate_id"] = "G0"
    result = evaluate_evidence("G1", evidence_fixture, rule_fixture)
    assert any(c["metric"] == "gate_id" and c["status"] == BLOCKED for c in result["checks"])


def test_conflicting_top_level_stage_blocked(rule_fixture, evidence_fixture):
    evidence_fixture["stage"] = "G2"
    result = evaluate_evidence("G1", evidence_fixture, rule_fixture)
    assert any(c["metric"] == "gate_id" and c["status"] == BLOCKED for c in result["checks"])


def test_not_run_template_insufficient(rule_fixture):
    evidence = {"schema_version": "1.0", "gate_id": "G1", "source_type": "not_run",
                "metrics": {}, "provenance": {}, "artifact_refs": []}
    result = evaluate_evidence("G1", evidence, rule_fixture)
    assert result["status"] == INSUFFICIENT
    assert "metrics" in result["missing_fields"]
    assert result["evidence_sha256"]


def test_gpu_smoke_legacy_tag_needs_scope(rule_fixture, evidence_fixture):
    evidence_fixture["source_type"] = "measured_gpu"
    result = evaluate_evidence("G1", evidence_fixture, rule_fixture)
    assert any(c["metric"] == "scope" and c["status"] == INSUFFICIENT for c in result["checks"])
    evidence_fixture["scope"] = "math_reference"
    result = evaluate_evidence("G1", evidence_fixture, rule_fixture)
    assert any(c["metric"] == "scope" and c["status"] == INSUFFICIENT for c in result["checks"])


def test_unknown_stage_blocked(rule_fixture, evidence_fixture):
    assert evaluate_evidence("G9", evidence_fixture, rule_fixture)["status"] == BLOCKED


def test_bad_rules_fail_closed(rule_fixture):
    bad = copy.deepcopy(rule_fixture)
    bad["gates"][0]["routes"]["accuracy"][0]["op"] = "always_pass"
    with pytest.raises(ValueError):
        validate_rules(bad)
    bad = copy.deepcopy(rule_fixture)
    bad["gates"][0]["routes"]["accuracy"][0]["value"] = float("nan")
    with pytest.raises(ValueError):
        validate_rules(bad)


def test_invalid_json_report_is_blocked_and_hashed(tmp_path, rule_fixture):
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps(rule_fixture))
    evidence = tmp_path / "fixture_bad.json"
    evidence.write_text('{"source_type":"unit_test_fixture", "metrics":{"error":NaN}}')
    output = tmp_path / "blocked_fixture_report.json"
    result = evaluate_file("G1", evidence, output, rules)
    assert result["status"] == BLOCKED
    assert len(result["evidence_sha256"]) == 64
    assert json.loads(output.read_text())["status"] == BLOCKED


def test_duplicate_json_keys_blocked(tmp_path, rule_fixture):
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps(rule_fixture))
    evidence = tmp_path / "fixture_duplicate.json"
    evidence.write_text('{"gate_id":"G0","gate_id":"G1"}')
    result = evaluate_file("G1", evidence, rules_path=rules)
    assert result["status"] == BLOCKED
    assert "Duplicate JSON key" in result["checks"][0]["reason"]


def test_cli_not_run_returns_zero_but_require_pass_is_nonzero(tmp_path, rule_fixture):
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps(rule_fixture))
    evidence = tmp_path / "NOT_RUN.json"
    evidence.write_text(json.dumps({"schema_version": "1.0", "gate_id": "G1",
                                  "source_type": "not_run", "metrics": {},
                                  "provenance": {}, "artifact_refs": []}))
    output = tmp_path / "insufficient.json"
    argv = ["evaluate", "--stage", "G1", "--evidence", str(evidence),
            "--output", str(output), "--rules", str(rules)]
    assert main(argv) == 0
    assert json.loads(output.read_text())["status"] == INSUFFICIENT
    assert main(argv + ["--require-pass"]) == 2


def test_default_rules_and_all_not_run_templates():
    root = Path(__file__).resolve().parents[1]
    rules = load_rules(root / "configs/gate_rules.json")
    for stage in ("G0", "G1", "G2", "G3", "G4"):
        path = root / "examples" / f"{stage}_evidence_NOT_RUN.json"
        evidence = json.loads(path.read_text(encoding="utf-8"))
        result = evaluate_evidence(stage, evidence, rules)
        assert result["status"] == INSUFFICIENT
        assert not result["passing_routes"]


def test_pure_engine_never_writes_a_fixture_evidence_artifact(gate_fixture, tmp_path):
    before = list(tmp_path.iterdir())
    result = evaluate_metrics(gate_fixture, {"correct": True, "error": 0.0, "gain": 5.0})
    assert result["status"] == PASS
    assert list(tmp_path.iterdir()) == before
