"""Portable, dependency-light notebook runtime utilities.

Only explicit configuration is recorded. Tokens/environment secrets are never
enumerated. A successful CPU preflight is not GPU/backend compatibility proof.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False,
                      default=_json_default).encode("utf-8")


def config_hash(config: dict) -> str:
    return hashlib.sha256(canonical_json(config)).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                         allow_nan=False, default=_json_default) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_report(*, probe_gpu: bool = True) -> dict:
    report = {
        "schema_version": "1.0", "created_at_utc": utc_now(),
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": {name: _package_version(name) for name in [
            "numpy", "scipy", "torch", "transformers", "datasets", "accelerate",
            "safetensors", "nbformat", "nbclient", "fast-hadamard-transform",
            "flash-attn", "triton", "lm_eval", "tokenizers", "huggingface-hub", "pyarrow", "psutil"]},
        "cuda": {"available": False, "status": "NOT_PROBED"},
        "compatibility_claim": "CPU tooling only; GPU kernels require target-device tests",
    }
    if not probe_gpu:
        return report
    if importlib.util.find_spec("torch") is None:
        report["cuda"]["status"] = "SKIPPED_TORCH_NOT_INSTALLED"
        return report
    import torch
    report["cuda"] = {
        "available": bool(torch.cuda.is_available()),
        "torch_cuda_build": torch.version.cuda,
        "status": "CPU_ONLY" if not torch.cuda.is_available() else "DEVICE_DETECTED",
    }
    if torch.cuda.is_available():
        report["cuda"]["devices"] = [
            {"index": i, "name": torch.cuda.get_device_name(i),
             "total_gib": torch.cuda.get_device_properties(i).total_memory / 2**30,
             "compute_capability": list(torch.cuda.get_device_capability(i))}
            for i in range(torch.cuda.device_count())]
        report["cuda"]["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        try:
            cp = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                 "--format=csv,noheader"],
                text=True, capture_output=True, timeout=15, check=False)
            report["cuda"]["nvidia_smi"] = cp.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            report["cuda"]["nvidia_smi"] = "unavailable"
    return report


def resolve_paths(project_root: str | Path | None = None,
                  run_root: str | Path | None = None,
                  durable_root: str | Path | None = None) -> dict[str, Path | None]:
    root = Path(project_root or os.environ.get("CELLQUANT_PROJECT_ROOT", Path.cwd())).resolve()
    if not (root / "pyproject.toml").exists():
        candidates = [root.parent, Path("/content/cellquant_research_kit"),
                      Path("/root/autodl-tmp/cellquant_research_kit")]
        found = [p for p in candidates if (p / "pyproject.toml").exists()]
        if not found:
            raise FileNotFoundError("Set PROJECT_ROOT to the extracted cellquant_research_kit directory")
        root = found[0].resolve()
    runs = Path(run_root or os.environ.get("CELLQUANT_RUN_ROOT", root / "runs")).resolve()
    durable = Path(durable_root).resolve() if durable_root else None
    runs.mkdir(parents=True, exist_ok=True)
    return {"project": root, "runs": runs, "durable": durable}


def build_manifest(root: str | Path, files: Sequence[str | Path], *,
                   config: dict | None = None, run_kind: str = "tooling") -> dict:
    root = Path(root).resolve()
    records = []
    for value in sorted(set(map(str, files))):
        path = Path(value)
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        relative = path.relative_to(root)  # reject references outside this bundle
        if not path.is_file():
            raise FileNotFoundError(f"Requested manifest artifact does not exist: {relative}")
        records.append({"path": relative.as_posix(), "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path)})
    return {"schema_version": "1.0", "created_at_utc": utc_now(),
            "run_kind": run_kind, "config": config or {},
            "config_sha256": config_hash(config or {}), "files": records}


def verify_manifest(root: str | Path, manifest: dict) -> dict:
    root = Path(root).resolve()
    missing, changed = [], []
    for rec in manifest.get("files", []):
        path = (root / rec["path"]).resolve()
        if not path.is_relative_to(root):
            changed.append(rec["path"] + ": outside root")
        elif not path.is_file():
            missing.append(rec["path"])
        elif path.stat().st_size != rec["size_bytes"] or sha256_file(path) != rec["sha256"]:
            changed.append(rec["path"])
    config_ok = config_hash(manifest.get("config", {})) == manifest.get("config_sha256")
    if not manifest.get("files"):
        return {"status": "INCOMPLETE", "missing_files": [], "changed_files": [],
                "config_hash_matches": config_ok, "scope": "No artifacts were checked"}
    return {"status": "PASS" if not missing and not changed and config_ok else "FAIL",
            "missing_files": missing, "changed_files": changed, "config_hash_matches": config_ok,
            "scope": "Artifact integrity only; not model accuracy or hardware compatibility"}


def extract_project_archive(archive: str | Path, destination: str | Path,
                            *, expected_sha256: str | None = None,
                            max_uncompressed_bytes: int = 2 * 2**30) -> Path:
    archive, destination = Path(archive), Path(destination).resolve()
    if expected_sha256 and sha256_file(archive) != expected_sha256:
        raise ValueError("Archive SHA-256 mismatch")
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if sum(i.file_size for i in infos) > max_uncompressed_bytes:
            raise ValueError("Archive exceeds extraction budget")
        for info in infos:
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive path or symbolic link")
        destination.mkdir(parents=True, exist_ok=True)
        z.extractall(destination)
    return destination


def copy_completed_run(source: str | Path, destination: str | Path) -> dict:
    """Copy a completed small run, excluding model caches/checkpoints by design."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Destination must be outside the source run tree")
    allowed = {".json", ".jsonl", ".csv", ".tsv", ".md", ".txt", ".log", ".png", ".pdf", ".npz"}
    copied, skipped = [], []
    small_checkpoints = {"R.bin", "rotation.bin", "rotation_state.pt", "rotation_optimizer.pt",
                         "rotation_checkpoint.pt", "calibration_state.pt"}
    for src in sorted(source.rglob("*")):
        if not src.is_file():
            continue
        if src.name == "migration_manifest.json":
            skipped.append({"path": src.relative_to(source).as_posix(), "reason": "superseded by a new destination manifest"})
            continue
        if (src.suffix not in allowed and src.name not in small_checkpoints) or src.name.startswith("."):
            skipped.append({"path": src.relative_to(source).as_posix(), "reason": "not in explicit artifact/checkpoint allowlist"})
            continue
        if src.stat().st_size > 256 * 2**20:
            skipped.append({"path": src.relative_to(source).as_posix(), "reason": "exceeds 256 MiB small-run transfer limit; transfer explicitly and hash"})
            continue
        rel = src.relative_to(source)
        dest = destination / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_name(dest.name + ".partial")
        shutil.copy2(src, temporary)
        os.replace(temporary, dest)
        copied.append(rel.as_posix())
    manifest = build_manifest(destination, copied, run_kind="migration_bundle")
    atomic_json(destination / "migration_manifest.json", manifest)
    return {"status": "COPIED" if copied else "NO_FILES_COPIED", "files": len(copied),
            "manifest": str(destination / "migration_manifest.json"), "skipped": skipped}


def stream_command(command: Sequence[str], log_path: str | Path, *,
                   timeout_s: float = 3600, poll_interval_s: float = 0.2,
                   cwd: str | Path | None = None, echo: bool = True) -> dict:
    """Nonblocking, interruptible log tailing. No shell, no stdout.readline loop."""
    if isinstance(command, str):
        raise TypeError("Pass an argv list; shell execution is disabled")
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    stopped = None
    with path.open("wb") as output, path.open("rb") as reader:
        process = subprocess.Popen(list(command), cwd=cwd, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=(os.name == "posix"))
        try:
            heartbeat_time = 0.0
            while True:
                if time.monotonic() - heartbeat_time >= 5.0:
                    atomic_json(path.with_suffix(path.suffix + '.heartbeat.json'), {
                        'pid': process.pid, 'status': 'RUNNING', 'updated_at_utc': utc_now(),
                        'elapsed_seconds': time.monotonic() - start})
                    heartbeat_time = time.monotonic()
                chunk = reader.read(65536)
                if chunk and echo:
                    print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
                rc = process.poll()
                if rc is not None:
                    remainder = reader.read()
                    if remainder and echo:
                        print(remainder.decode("utf-8", errors="replace"), end="", flush=True)
                    break
                if time.monotonic() - start > timeout_s:
                    stopped = "TIMEOUT"
                    raise TimeoutError(f"Command exceeded {timeout_s:g} seconds; log: {path}")
                time.sleep(poll_interval_s)
        except BaseException:
            atomic_json(path.with_suffix(path.suffix + '.heartbeat.json'), {
                'pid': process.pid, 'status': stopped or 'INTERRUPTED_OR_FAILED', 'updated_at_utc': utc_now()})
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=5)
            raise
    atomic_json(path.with_suffix(path.suffix + '.heartbeat.json'), {
        'pid': process.pid, 'status': 'FINISHED' if rc == 0 else 'PROCESS_FAILED',
        'returncode': rc, 'updated_at_utc': utc_now()})
    return {"returncode": rc, "elapsed_seconds": time.monotonic() - start,
            "status": "PROCESS_FINISHED" if rc == 0 else "PROCESS_FAILED",
            "log_path": str(path), "stop_reason": stopped,
            "note": "A zero exit code is not scientific evidence; validate result artifacts separately"}


def gpu_kernel_probe(*, test_optional_fht: bool = True) -> dict:
    env = environment_report()
    if not env["cuda"].get("available"):
        return {"status": "SKIPPED", "environment": env,
                "reason": "No CUDA device; target GPU compatibility remains untested"}
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.manual_seed(20261115)
    try:
        a = torch.randn(64, 128, device="cuda", dtype=dtype)
        b = torch.randn(128, 64, device="cuda", dtype=dtype)
        y = a @ b
        reference = a.float() @ b.float()
        relative = float((y.float() - reference).norm() / reference.norm())
    except Exception as exc:
        return {"status": "FAIL", "environment": env, "phase": "torch_cuda_gemm",
                "error_type": type(exc).__name__, "message": str(exc)[:1000],
                "scope": "Target CUDA stack cannot execute the small GEMM probe"}
    fht = {"status": "NOT_TESTED"}
    if test_optional_fht:
        if importlib.util.find_spec("fast_hadamard_transform") is None:
            fht = {"status": "MISSING", "reason": "Compile/install against this torch/CUDA/GPU before claiming readiness"}
        else:
            try:
                from fast_hadamard_transform import hadamard_transform
                x = torch.randn(4, 128, device="cuda", dtype=torch.float32)
                y1 = hadamard_transform(x, scale=128**-0.5)
                y2 = hadamard_transform(y1, scale=128**-0.5)
                error = float((y2 - x).norm() / x.norm())
                torch.cuda.synchronize()
                fht = {"status": "PASS" if error < 1e-5 else "FAIL", "inverse_relative_error": error}
            except Exception as exc:
                fht = {"status": "FAIL", "error_type": type(exc).__name__, "message": str(exc)[:1000]}
    torch.cuda.synchronize()
    optional={}
    for name in ['flash_attn','triton']:
        if importlib.util.find_spec(name) is None:
            optional[name]={'status':'MISSING'};continue
        try:
            if name=='triton':
                from .triton_probe import probe
                optional[name]=probe()
            else:
                from flash_attn import flash_attn_func
                q=torch.randn(1,16,2,32,device='cuda',dtype=dtype);k=torch.randn_like(q);v=torch.randn_like(q)
                actual=flash_attn_func(q,k,v,dropout_p=0.,causal=True)
                ref=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2).float(),k.transpose(1,2).float(),v.transpose(1,2).float(),is_causal=True).transpose(1,2)
                error=float((actual.float()-ref).norm()/ref.norm());torch.cuda.synchronize()
                optional[name]={'status':'PASS' if error<.02 else 'FAIL','relative_error':error,'scope':'Small causal attention forward only'}
        except Exception as exc:
            optional[name]={'status':'FAIL','error_type':type(exc).__name__,'message':str(exc)[:1000]}
    return {"status": "PASS" if relative < 0.02 and fht["status"] == "PASS" else "INCOMPLETE",
            "environment": env, "dtype": str(dtype), "gemm_relative_error": relative,
            "fht": fht, "optional_kernels":optional,"scope": "Small CUDA/FHT/optional-kernel smoke only; not full SpinQuant/model/GPTQ compatibility"}
