"""Fail-closed research-stage gates.

This module summarizes supplied evidence. It cannot establish that measurements
or provenance statements are truthful. A PASS is a project decision, never a
conference acceptance prediction. Synthetic unit-test fixtures are always
blocked at the evidence boundary, even if custom rules allow their source type.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

PASS = "PASS"
FAIL = "FAIL"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
BLOCKED = "BLOCKED"
SCHEMA_VERSION = "1.0"
_ABSENT = object()
_LEGACY_SCOPES = {
    "G0": "math_reference",
    "G1": "real_layer_panel",
    "G2": "full_model",
    "G3": "full_model",
    "G4": "full_model",
}
_FORBIDDEN_SOURCES = {
    "published", "paper_reported", "estimate", "estimated",
    "unit_test_fixture", "mock", "simulated_result",
}


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _safe_observed(value: Any) -> Any:
    """Reports must themselves be strict JSON, including for bad measurements."""
    if value is _ABSENT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return repr(value)


def _check(condition: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    name = condition["metric"]
    expected = condition["value"]
    actual = metrics.get(name, _ABSENT)
    result = {
        "metric": name,
        "op": condition["op"],
        "expected": expected,
        "observed": _safe_observed(actual),
        "unit": condition.get("unit", ""),
        "rationale": condition.get("rationale", ""),
    }
    if actual is _ABSENT or actual is None:
        return dict(result, status=INSUFFICIENT, reason="Missing measurement; no default value was substituted.")
    if isinstance(expected, bool):
        valid_type = isinstance(actual, bool)
    elif _finite_number(expected):
        valid_type = _finite_number(actual)
    elif isinstance(expected, str):
        valid_type = isinstance(actual, str)
    else:
        valid_type = False
    if not valid_type:
        return dict(result, status=INSUFFICIENT,
                    reason="Wrong type or non-finite measurement; values are not coerced.")
    op = condition["op"]
    if op == "eq":
        passed = actual == expected
    elif op == "ge":
        passed = actual >= expected
    elif op == "le":
        passed = actual <= expected
    else:
        raise ValueError(f"Unsupported comparator: {op!r}")
    return dict(result, status=PASS if passed else FAIL,
                reason="Threshold satisfied." if passed else "Measured value does not satisfy the frozen threshold.")


def _all_status(checks: list[dict[str, Any]]) -> str:
    # A definite failed mandatory check remains failed even if other data are absent.
    for state in (BLOCKED, FAIL, INSUFFICIENT):
        if any(check["status"] == state for check in checks):
            return state
    return PASS


def evaluate_metrics(gate: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Pure threshold evaluator, not an experiment-evidence certificate.

    This function is intentionally separate so tests can exercise numeric logic
    without manufacturing a PASS research evidence document.
    """
    common = [_check(c, metrics) for c in gate["common_checks"]]
    routes: dict[str, Any] = {}
    for name, conditions in gate["routes"].items():
        checks = [_check(c, metrics) for c in conditions]
        routes[name] = {
            "status": _all_status(checks),
            "checks": checks,
            "missing_fields": [f"metrics.{c['metric']}" for c in checks if c["status"] == INSUFFICIENT],
        }
    common_state = _all_status(common)
    if common_state != PASS:
        state = common_state
    elif any(route["status"] == PASS for route in routes.values()):
        state = PASS
    elif any(route["status"] == INSUFFICIENT for route in routes.values()):
        state = INSUFFICIENT
    else:
        state = FAIL
    missing = [f"metrics.{c['metric']}" for c in common if c["status"] == INSUFFICIENT]
    for route in routes.values():
        missing.extend(route["missing_fields"])
    return {
        "status": state,
        "checks": common,
        "routes": routes,
        "missing_fields": list(dict.fromkeys(missing)),
        "passing_routes": [name for name, route in routes.items() if route["status"] == PASS],
    }


def _meta_check(name: str, status: str, reason: str, observed: Any = None) -> dict[str, Any]:
    return {"metric": name, "op": "metadata", "expected": None,
            "observed": _safe_observed(observed), "unit": "metadata",
            "rationale": reason, "status": status, "reason": reason}


def _canonical_digest(obj: Any) -> str:
    # allow_nan here is only to hash an in-memory invalid fixture deterministically.
    # File ingestion rejects non-standard JSON constants; output never contains them.
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def evaluate_evidence(
    stage: str,
    evidence: Mapping[str, Any],
    rules: Mapping[str, Any],
    *,
    evidence_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate one evidence document, checking stage and measurement scope first."""
    validate_rules(rules)
    gate = next((g for g in rules["gates"] if g["gate_id"] == stage), None)
    digest = evidence_sha256 or _canonical_digest(evidence)
    if gate is None:
        return _blocked_report(stage, f"Unknown stage {stage!r}.", digest)
    if not isinstance(evidence, Mapping):
        return _blocked_report(stage, "Evidence must be a JSON object.", digest)

    metadata: list[dict[str, Any]] = []
    schema = evidence.get("schema_version")
    metadata.append(_meta_check("schema_version", PASS if schema == SCHEMA_VERSION else BLOCKED,
                                "Evidence schema_version must be the string '1.0'.", schema))
    ids = [evidence[key] for key in ("gate_id", "stage") if key in evidence]
    if not ids:
        metadata.append(_meta_check("gate_id", INSUFFICIENT, "Evidence must identify its gate_id or stage."))
    else:
        metadata.append(_meta_check("gate_id", PASS if all(value == stage for value in ids) else BLOCKED,
                                    "All supplied stage identifiers must match the requested stage.", ids))

    source = evidence.get("source_type")
    provenance = evidence.get("provenance")
    if source is None or source == "not_run":
        metadata.append(_meta_check("source_type", INSUFFICIENT,
                                    "Not-run templates and missing source types contain no experiment evidence.", source))
    elif not isinstance(source, str):
        metadata.append(_meta_check("source_type", BLOCKED, "source_type must be a string.", source))
    elif source in _FORBIDDEN_SOURCES:
        metadata.append(_meta_check("source_type", BLOCKED,
                                    "Published, estimated, mock, or unit-test data cannot pass an experiment gate.", source))
    elif source not in gate["allowed_source_types"]:
        metadata.append(_meta_check("source_type", BLOCKED,
                                    "Evidence source does not match this stage's allowed measurement scope.", source))
    else:
        metadata.append(_meta_check("source_type", PASS, "Allowed measurement source; truthfulness is not automatically certified.", source))

    if isinstance(source, str) and source in ("measured_cpu", "measured_gpu"):
        scope = evidence.get("scope")
        expected_scope = _LEGACY_SCOPES.get(stage)
        metadata.append(_meta_check("scope", PASS if scope == expected_scope else INSUFFICIENT,
                                    f"Legacy measurement tags require scope={expected_scope!r}; GPU smoke is not a real-model study.", scope))

    if not isinstance(provenance, Mapping) or not provenance:
        metadata.append(_meta_check("provenance", INSUFFICIENT,
                                    "Provide nonempty provenance generated by the measurement pipeline."))
    else:
        fixture_tags = [provenance.get(key) for key in ("kind", "source_type", "evidence_kind")]
        if "unit_test_fixture" in fixture_tags or provenance.get("unit_test_fixture") is True:
            metadata.append(_meta_check("provenance", BLOCKED,
                                        "A unit-test fixture is never promotable to research evidence."))
        else:
            matching = all(provenance[key] == stage for key in ("gate_id", "stage") if key in provenance)
            metadata.append(_meta_check("provenance", PASS if matching else BLOCKED,
                                        "Any provenance stage identifier must agree; model/code/protocol references require audit."))
    refs = evidence.get("artifact_refs")
    if not isinstance(refs, list) or not refs:
        metadata.append(_meta_check("artifact_refs", INSUFFICIENT,
                                    "Reference original measurement artifacts; the engine does not independently authenticate them."))
    elif any(not isinstance(ref, (str, Mapping)) or not ref for ref in refs):
        metadata.append(_meta_check("artifact_refs", BLOCKED, "Artifact references must be nonempty strings or objects."))
    else:
        metadata.append(_meta_check("artifact_refs", PASS,
                                    "Artifact references supplied; existence, hashes, and provenance remain pipeline/audit responsibilities."))

    metrics = evidence.get("metrics")
    if not isinstance(metrics, Mapping):
        metadata.append(_meta_check("metrics", BLOCKED, "metrics must be a JSON object; arrays and strings are invalid."))
        metrics = {}
    elif not metrics:
        metadata.append(_meta_check("metrics", INSUFFICIENT, "An empty metrics object has no measurements."))

    numeric = evaluate_metrics(gate, metrics)
    meta_state = _all_status(metadata)
    state = meta_state if meta_state != PASS else numeric["status"]
    missing = [check["metric"] for check in metadata if check["status"] == INSUFFICIENT]
    missing.extend(numeric["missing_fields"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate_id": stage,
        "status": state,
        "routes": numeric["routes"],
        "checks": metadata + numeric["checks"],
        "missing_fields": list(dict.fromkeys(missing)),
        "evidence_sha256": digest,
        "decision_scope": gate["on_pass"] if state == PASS else "No promotion authorized by this report.",
        "passing_routes": numeric["passing_routes"] if state == PASS else [],
        "evidence_source_type": source,
        "disclaimer": "Threshold aggregation only: this engine does not certify evidence truthfulness, novelty, or conference acceptance.",
    }
    if state == PASS:
        report["nonblocking_missing_fields"] = numeric["missing_fields"]
    return report


def validate_rules(rules: Mapping[str, Any]) -> None:
    if not isinstance(rules, Mapping) or rules.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Rules schema_version must be '1.0'.")
    gates = rules.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("Rules must contain a nonempty gates list.")
    ids: set[str] = set()
    for gate in gates:
        if not isinstance(gate, Mapping) or not isinstance(gate.get("gate_id"), str) or not gate["gate_id"]:
            raise ValueError("Each rule gate requires a gate_id.")
        if gate["gate_id"] in ids:
            raise ValueError("Duplicate rule gate_id.")
        ids.add(gate["gate_id"])
        if not isinstance(gate.get("on_pass"), str):
            raise ValueError("Each gate requires on_pass decision scope.")
        if not isinstance(gate.get("allowed_source_types"), list) or not gate["allowed_source_types"]:
            raise ValueError("Each gate requires explicit allowed_source_types.")
        common = gate.get("common_checks")
        routes = gate.get("routes")
        if not isinstance(common, list) or not isinstance(routes, Mapping) or not routes:
            raise ValueError("Each gate requires common_checks and nonempty routes.")
        conditions = list(common)
        for name, route in routes.items():
            if not isinstance(name, str) or not isinstance(route, list) or not route:
                raise ValueError("Each route requires a name and nonempty condition list.")
            conditions.extend(route)
        for condition in conditions:
            if not isinstance(condition, Mapping) or not isinstance(condition.get("metric"), str):
                raise ValueError("Each condition requires a metric name.")
            op = condition.get("op")
            value = condition.get("value")
            if op not in ("eq", "ge", "le"):
                raise ValueError("Unknown comparator in rules.")
            if not (isinstance(value, (bool, str)) or _finite_number(value)):
                raise ValueError("Threshold values must be finite numbers, booleans, or strings.")
            if op in ("ge", "le") and not _finite_number(value):
                raise ValueError("Ordered comparisons require finite numeric thresholds.")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-standard non-finite JSON constant: {value}")


def _parse_json(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs_no_duplicates, parse_constant=_reject_constant)


def resolve_rules_path(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    package_root = Path(__file__).resolve().parents[2]
    candidates = [Path.cwd() / "configs/gate_rules.json", package_root / "configs/gate_rules.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate configs/gate_rules.json; supply --rules PATH.")


def load_rules(path: str | Path | None = None) -> dict[str, Any]:
    rules = _parse_json(resolve_rules_path(path).read_bytes())
    validate_rules(rules)
    return rules


def _blocked_report(stage: str, reason: str, digest: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "gate_id": stage, "status": BLOCKED,
        "routes": {}, "checks": [_meta_check("evidence_document", BLOCKED, reason)],
        "missing_fields": [], "evidence_sha256": digest,
        "decision_scope": "No promotion authorized by this report.",
        "passing_routes": [],
        "disclaimer": "Threshold aggregation does not authenticate experiment evidence.",
    }


def write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent,
                                         prefix=output.name + ".", suffix=".tmp", delete=False) as handle:
            temp_name = handle.name
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, output)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def evaluate_file(
    stage: str,
    evidence_path: str | Path,
    output_path: str | Path | None = None,
    rules_path: str | Path | None = None,
) -> dict[str, Any]:
    digest = None
    try:
        raw = Path(evidence_path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        evidence = _parse_json(raw)
        path = resolve_rules_path(rules_path)
        rules_raw = path.read_bytes()
        rules = _parse_json(rules_raw)
        report = evaluate_evidence(stage, evidence, rules, evidence_sha256=digest)
        report["rules_sha256"] = hashlib.sha256(rules_raw).hexdigest()
        report["rules_path"] = str(path)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
        report = _blocked_report(stage, f"{type(exc).__name__}: {exc}", digest)
    if output_path is not None:
        write_report(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate one stage without inventing missing measurements.")
    evaluate.add_argument("--stage", required=True)
    evaluate.add_argument("--evidence", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--rules")
    evaluate.add_argument("--require-pass", action="store_true",
                          help="Return exit code 2 unless status is PASS. Default exit 0 is for notebook inspection only.")
    args = parser.parse_args(argv)
    report = evaluate_file(args.stage, args.evidence, args.output, args.rules)
    print(json.dumps({"gate_id": report["gate_id"], "status": report["status"],
                      "decision_scope": report["decision_scope"], "report_path": str(Path(args.output).resolve())},
                     ensure_ascii=False))
    return 2 if args.require_pass and report["status"] != PASS else 0


if __name__ == "__main__":
    raise SystemExit(main())
