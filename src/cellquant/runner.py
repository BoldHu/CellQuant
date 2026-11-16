"""Run planning, evidence guards, resumable process launching, real-result summary.

The global-R1/GPTQ experiment backend is provided by cellquant.experiment.
It refuses to turn smoke checks or missing evidence into permission to run a
large experiment matrix. Threshold logic lives in cellquant.gates.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import time
import re
import subprocess
from pathlib import Path

from .runtime import atomic_json, config_hash, read_json, sha256_file, stream_command, utc_now, environment_report


def backend_available(module="cellquant.experiment") -> bool:
    try:
        if importlib.util.find_spec(module) is None:
            return False
        backend = importlib.import_module(module)
        return (getattr(backend, "BACKEND_CONTRACT_VERSION", None) == "1.0"
                and callable(getattr(backend, "run_experiment", None)))
    except (ImportError, ValueError, RuntimeError):
        return False


def frozen_execution_config_hash(config):
    return config_hash({k:v for k,v in config.items() if k not in {'resume_run_dir','result_path'}})


def execution_fingerprint(project_root, provenance_path=None):
    """Bind resume to source, active environment and resolved model/data inputs."""
    root = Path(project_root).resolve()
    source_files = []
    for subdir in ["src", "scripts", "requirements"]:
        source_files.extend(p for p in (root / subdir).rglob("*")
                            if p.is_file() and p.suffix in {".py", ".txt", ".toml"}
                            and not any(part.endswith(('.egg-info','.dist-info')) or part=='__pycache__' for part in p.parts))
    if (root / "pyproject.toml").is_file():
        source_files.append(root / "pyproject.toml")
    source = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(source_files)}
    code_git={'commit':None,'tracked_clean':None,'untracked_source_files':[]}
    if (root/'.git').exists():
        base=['git','-c',f'safe.directory={root.as_posix()}','-C',str(root)]
        cp=subprocess.run(base+['rev-parse','HEAD'],capture_output=True,text=True,check=False)
        if cp.returncode==0:
            code_git['commit']=cp.stdout.strip()
            code_git['tracked_clean']=not subprocess.check_output(base+['status','--porcelain','--untracked-files=no'],text=True).strip()
            tracked=set(subprocess.check_output(base+['ls-files'],text=True).splitlines())
            code_git['untracked_source_files']=sorted(set(source)-tracked)
    env = environment_report()
    stable_env = {k: env[k] for k in ["python", "platform", "machine", "packages", "cuda"]}
    missing, provenance = [], None
    if not source:
        missing.append("No source files for execution fingerprint")
    if not provenance_path or not Path(provenance_path).is_file():
        missing.append("Verified model/tokenizer/data/protocol provenance file missing")
    else:
        path = Path(provenance_path).resolve()
        provenance = read_json(path)
        models = provenance.get("models", {})
        if not models:
            missing.append("Resolved model/tokenizer revisions missing")
        for model_id, revisions in models.items():
            for field in ["model_revision", "tokenizer_revision"]:
                revision = revisions.get(field, "")
                if not re.fullmatch(r"[a-fA-F0-9]{40}|[a-fA-F0-9]{64}", revision):
                    missing.append(f"Immutable revision required: {model_id}/{field}")
        for artifact_key, hash_key in [("data_manifest_path", "data_manifest_sha256"),
                                       ("protocol_manifest_path", "protocol_sha256")]:
            value = provenance.get(artifact_key)
            target = Path(value) if value else None
            if target is not None and not target.is_absolute():
                target = root / target
            if target is None or not target.is_file():
                missing.append(f"Missing provenance artifact: {artifact_key}")
            elif sha256_file(target) != provenance.get(hash_key):
                missing.append(f"Provenance artifact hash mismatch: {artifact_key}")
            elif artifact_key=='data_manifest_path':
                data=read_json(target)
                if data.get('kind')=='token_manifest_bundle':
                    for model,records in data['models'].items():
                        for seed,ref in records.items():
                            p=Path(ref['path'])
                            if not p.is_file() or sha256_file(p)!=ref['sha256']:
                                missing.append(f'Token bundle artifact mismatch: {model}/{seed}')
                            else:
                                token=read_json(p);cache=p.parent/token['tokens_path']
                                if not cache.is_file() or sha256_file(cache)!=token['tokens_sha256']:
                                    missing.append(f'Token cache checksum mismatch: {model}/{seed}')
                elif data.get('tokens_path'):
                    cache=target.parent/data['tokens_path']
                    if not cache.is_file() or sha256_file(cache)!=data.get('tokens_sha256'):
                        missing.append('Token cache checksum mismatch')
        if provenance.get("dtype") not in {"bfloat16", "float16", "float32", "float64"}:
            missing.append("Explicit model dtype missing from execution provenance")
    payload = {"source_sha256": config_hash(source), "source_files": source,"code_git":code_git,
               "environment": stable_env, "provenance": provenance,
               "provenance_file_sha256": sha256_file(provenance_path)
               if provenance_path and Path(provenance_path).is_file() else None}
    return {"status": "VERIFIED" if not missing else "INCOMPLETE", "missing": missing,
            "sha256": config_hash(payload), "payload": payload}


def make_job_manifest(jobs, *, project_root, gpu_hour_budget=8.0,
                      required_gate="G3", backend_module="cellquant.experiment",
                      execution_provenance_path=None):
    root = Path(project_root).resolve()
    records = []
    ids = set()
    for raw in jobs:
        job = dict(raw)
        if job["job_id"] in ids:
            raise ValueError("Duplicate job_id")
        ids.add(job["job_id"])
        path = Path(job["config_path"])
        path = path if path.is_absolute() else root / path
        job["config_path"] = str(path.resolve())
        job["config_sha256"] = sha256_file(path) if path.is_file() else None
        job.setdefault("command", [sys.executable, "-m", backend_module, "--config", str(path.resolve())])
        job.setdefault("result_path", str(root / "runs" / job["job_id"] / "result.json"))
        records.append(job)
    provenance_path = Path(execution_provenance_path) if execution_provenance_path else root / "runs/provenance/execution_provenance.json"
    if not provenance_path.is_absolute():
        provenance_path = root / provenance_path
    fingerprint = execution_fingerprint(root, provenance_path)
    result = {"schema_version": "1.0", "created_at_utc": utc_now(),
              "status": "PLANNED_ONLY", "required_gate": required_gate,
              "gpu_hour_budget": float(gpu_hour_budget), "gpu_count": 1,
              "backend_module": backend_module, "jobs": records,
              "project_root": str(root), "execution_provenance_path": str(provenance_path),
              "execution_fingerprint": fingerprint,
              "note": "No model run was performed while constructing this manifest"}
    result["plan_sha256"] = config_hash({k: result[k] for k in [
        "required_gate", "gpu_hour_budget", "gpu_count", "backend_module", "jobs", "execution_fingerprint"]})
    return result


def check_execution_readiness(manifest, *, gate_path, gate_evidence_path=None):
    missing = []
    if not backend_available(manifest.get("backend_module", "cellquant.experiment")):
        missing.append("Experiment backend missing callable run_experiment / BACKEND_CONTRACT_VERSION='1.0'")
    if not manifest.get("jobs"):
        missing.append("No experiment jobs were specified")
    current = execution_fingerprint(manifest["project_root"], manifest.get("execution_provenance_path"))
    missing.extend(current["missing"])
    expected = manifest.get("execution_fingerprint", {})
    if expected.get("status") != "VERIFIED":
        missing.append("Manifest was planned without verified execution provenance")
    elif current["sha256"] != expected.get("sha256"):
        missing.append("Source/model/data/protocol/environment changed since planning; create and validate a new plan")
    gate_path = Path(gate_path)
    if not gate_path.is_file():
        missing.append("Required gate report is missing")
    else:
        try:
            gate = read_json(gate_path)
            if gate.get("gate_id") != manifest.get("required_gate") or gate.get("status") != "PASS":
                missing.append("Required gate has not passed")
            if not gate_evidence_path or not Path(gate_evidence_path).is_file():
                missing.append("Underlying gate evidence file is missing")
            elif sha256_file(gate_evidence_path) != gate.get("evidence_sha256"):
                missing.append("Gate evidence hash mismatch")
            else:
                # Recompute the gate using current rules, rather than trusting a
                # hand-edited PASS or a report produced before rules changed.
                from .gates import evaluate_file
                recomputed = evaluate_file(manifest["required_gate"], gate_evidence_path,
                                           rules_path=Path(manifest["project_root"]) / "configs/gate_rules.json")
                if recomputed.get("status") != "PASS":
                    missing.append("Current gate evaluator does not pass the underlying evidence")
                if not gate.get("rules_sha256") or gate.get("rules_sha256") != recomputed.get("rules_sha256"):
                    missing.append("Gate rules changed or original rules hash is absent")
                evidence = read_json(gate_evidence_path)
                if manifest.get("required_gate") == "G3":
                    provenance = evidence.get("provenance", {})
                    bound = provenance.get("execution_fingerprint_sha256") if isinstance(provenance, dict) else None
                    if bound != current["sha256"]:
                        missing.append("G3 is not bound to the current source/model/data/GPU/driver/dependency fingerprint")
        except (ValueError, TypeError, KeyError, OSError) as exc:
            missing.append("Gate evidence cannot be verified: " + type(exc).__name__)
    for job in manifest["jobs"]:
        path = Path(job["config_path"])
        if not path.is_file():
            missing.append(f"Config missing: {job['job_id']}")
        elif sha256_file(path) != job.get("config_sha256"):
            missing.append(f"Config changed since manifest: {job['job_id']}")
    return {"status": "READY" if not missing else "BLOCKED", "missing": missing,
            "scope": "Execution prerequisites only; the gate evaluator owns scientific acceptance"}


def execute_manifest(manifest, *, project_root, gate_path, gate_evidence_path,
                     state_dir, allow_execution=False, echo=True):
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    readiness = check_execution_readiness(manifest, gate_path=gate_path,
                                         gate_evidence_path=gate_evidence_path)
    atomic_json(state_dir / "readiness.json", readiness)
    if not allow_execution:
        return {"status": "MANIFEST_ONLY", "readiness": readiness, "jobs_executed": 0}
    if readiness["status"] != "READY":
        return {"status": "BLOCKED", "readiness": readiness, "jobs_executed": 0}
    budget = float(manifest["gpu_hour_budget"]) * 3600 / int(manifest.get("gpu_count", 1))
    start = time.monotonic()
    results = []
    for job in manifest["jobs"]:
        record_path = state_dir / f"{job['job_id']}.json"
        signature = config_hash({"job": job, "plan_sha256": manifest["plan_sha256"]})
        if record_path.is_file():
            prior = read_json(record_path)
            if prior.get("status") == "COMPLETE" and prior.get("signature") == signature:
                result_path = Path(job["result_path"])
                if result_path.is_file() and sha256_file(result_path) == prior.get("result_sha256"):
                    results.append({"job_id": job["job_id"], "status": "RESUMED_SKIP"})
                    continue
        remaining = budget - (time.monotonic() - start)
        if remaining <= 0:
            results.append({"job_id": job["job_id"], "status": "BUDGET_EXHAUSTED"})
            break
        record = {"job_id": job["job_id"], "status": "RUNNING", "signature": signature,
                  "started_at_utc": utc_now()}
        atomic_json(record_path, record)
        try:
            process_result = stream_command(job["command"], state_dir / f"{job['job_id']}.log",
                                            timeout_s=remaining, cwd=project_root, echo=echo)
            record["process"] = process_result
            artifact = Path(job["result_path"])
            if process_result["returncode"] != 0:
                record["status"] = "PROCESS_FAILED"
            elif not artifact.is_file():
                record["status"] = "MISSING_RESULT_ARTIFACT"
            else:
                payload = read_json(artifact)
                if payload.get("status") != "COMPLETE" or payload.get("evidence_kind") != "REAL_MODEL_EXPERIMENT":
                    record["status"] = "INVALID_RESULT_KIND"
                else:
                    record["status"] = "COMPLETE"
                    record["result_sha256"] = sha256_file(artifact)
        except KeyboardInterrupt:
            record["status"] = "INTERRUPTED"
            atomic_json(record_path, record)
            raise
        except TimeoutError as exc:
            record["status"] = "BUDGET_EXHAUSTED"
            record["message"] = str(exc)
        record["finished_at_utc"] = utc_now()
        atomic_json(record_path, record)
        results.append(record)
        if record["status"] != "COMPLETE":
            break
    final = {"status": "STOPPED" if any(r["status"] not in ["COMPLETE", "RESUMED_SKIP"] for r in results) else "FINISHED",
             "jobs": results, "wall_seconds": time.monotonic() - start,
             "note": "Evaluate G3/G4 from measured artifacts separately"}
    atomic_json(state_dir / "execution_summary.json", final)
    return final


def load_real_result_rows(paths):
    """No imputation, no paper numbers, no synthetic examples mixed into a table."""
    rows, rejected = [], []
    for value in paths:
        path = Path(value)
        record = read_json(path)
        reasons = []
        if record.get("evidence_kind") != "REAL_MODEL_EXPERIMENT":
            reasons.append("not a real model experiment")
        if record.get("status") != "COMPLETE":
            reasons.append("not complete")
        for field in ["model_id", "method", "seed", "protocol_sha256", "metrics"]:
            if field not in record:
                reasons.append(f"missing {field}")
        if reasons:
            rejected.append({"path": str(path), "reasons": reasons})
            continue
        row = {k: record[k] for k in ["model_id", "method", "seed", "protocol_sha256"]}
        row.update(record["metrics"])
        row["source_file"] = str(path)
        rows.append(row)
    return {"status": "AVAILABLE" if rows else "INSUFFICIENT_EVIDENCE",
            "rows": rows, "rejected": rejected,
            "note": "Cross-protocol or cross-paper rows must not be pooled as matched comparisons"}
