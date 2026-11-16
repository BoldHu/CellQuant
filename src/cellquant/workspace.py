"""Local execution with explicit, versioned Drive persistence (no credentials).

Snapshots store immutable content objects and commit a manifest last. They are
file backups, not scientific evidence or proof that a live checkpoint is safe.
Stop writers before backup; experiment resume still verifies its own provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .runtime import atomic_json, config_hash, read_json, sha256_file, utc_now

AREAS = {"models": ".cache/huggingface/hub", "datasets": ".cache/huggingface/datasets", "runs": "runs"}
EXCLUDED = {".git", ".locks", "__pycache__", "token", "stored_tokens", ".env"}


def workspace_environment(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("Expected the CellQuant checkout, containing pyproject.toml")
    hf = root / ".cache/huggingface"
    return {"CELLQUANT_PROJECT_ROOT": str(root), "CELLQUANT_RUN_ROOT": str(root / "runs"),
            "CELLQUANT_CACHE_DIR": str(hf / "hub"), "HF_HOME": str(hf),
            "HF_HUB_CACHE": str(hf / "hub"), "HF_DATASETS_CACHE": str(hf / "datasets"),
            "HF_TOKEN_PATH": str(hf / "token"), "PYTHONUTF8": "1"}


def git_identity(root: Path) -> dict:
    # This narrowly scoped trust is for the explicitly selected project only.
    prefix = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    def query(*args):
        return subprocess.run(prefix + list(args), check=True, capture_output=True,
                              text=True, timeout=20).stdout.strip()
    try:
        return {"commit": query("rev-parse", "HEAD"),
                "branch": query("branch", "--show-current"),
                "dirty": bool(query("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def configure_workspace(project_root: str | Path, drive_root: str | Path, *, apply: bool = True) -> dict:
    root, drive = Path(project_root).resolve(), Path(drive_root).resolve()
    env = workspace_environment(root)
    if drive == root or drive.is_relative_to(root) or root.is_relative_to(drive):
        raise ValueError("Keep the local checkout and durable storage in separate trees")
    # Never create an ordinary local folder where a missing Colab mount should be.
    if drive.as_posix().startswith("/content/drive/") and not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError(
            "Google Drive 尚未挂载到 /content/drive（MyDrive 入口不可用）。\n"
            "VS Code: Ctrl+Shift+P → Colab: Mount Google Drive to Server...，"
            "单独运行扩展追加到 notebook 末尾的单元并完成授权，再重跑当前单元。\n"
            "浏览器 Colab: 设置 MOUNT_DRIVE_IN_BROWSER=True 后重跑配置与当前单元。\n"
            f"项目文件夹 {drive.name!r} 会在挂载成功后自动创建，无需预先在云盘中新建。"
        )
    if not drive.parent.is_dir():
        raise FileNotFoundError(f"Durable parent is unavailable; mount/create it first: {drive.parent}")
    if apply and any(m in sys.modules for m in ("huggingface_hub", "datasets", "transformers")):
        if any(os.environ.get(k) != env[k] for k in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE")):
            raise RuntimeError("Restart the kernel and configure caches before importing Hugging Face packages")
    for relative in AREAS.values():
        (root / relative).mkdir(parents=True, exist_ok=True)
    drive.mkdir(exist_ok=True)
    env["CELLQUANT_DRIVE_ROOT"] = str(drive)
    if apply:
        os.environ.update(env)
    record = {"schema_version": "1.0", "created_at_utc": utc_now(), "environment": env,
              "code": git_identity(root), "source_type": "not_run", "status": "CONFIGURED",
              "note": "No models downloaded, experiments run, or scientific gates passed"}
    atomic_json(root / ".cache/colab_workspace.json", record)
    return record


def _paths(root, drive, area):
    if area not in AREAS:
        raise ValueError(f"Unknown area: {area}")
    root, drive = Path(root).resolve(), Path(drive).resolve()
    local = (root / AREAS[area]).resolve()
    if not local.is_relative_to(root):
        raise ValueError("Managed area must be local, not a link outside the checkout")
    if not drive.is_dir():
        raise FileNotFoundError("Run workspace init with the mounted Drive directory first")
    if drive == root or drive.is_relative_to(root) or root.is_relative_to(drive):
        raise ValueError("Local and durable directories must not overlap")
    return local, drive


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or not path.parts or path.is_absolute() or any(p in {"..", "."} for p in path.parts) or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe snapshot path: {value!r}")
    if path.as_posix() != value:
        raise ValueError(f"Non-canonical snapshot path: {value!r}")
    return path


def _excluded(path: Path) -> bool:
    return any(p in EXCLUDED or p.startswith(".env.") or p.endswith((".lock", ".partial", ".incomplete"))
               or p.startswith(".") for p in path.parts)


def _files(root: Path):
    for parent, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not _excluded((Path(parent) / d).relative_to(root)))
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                raise ValueError("Directory symlinks are not supported in workspace snapshots")
        for name in sorted(names):
            path = Path(parent) / name
            if not _excluded(path.relative_to(root)):
                yield path


def _signature(path):
    s = path.stat()
    return s.st_size, s.st_mtime_ns, s.st_ino


def _copy_verified(source: Path, destination: Path, expected: str, size: int):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".transfer-", suffix=".partial", dir=destination.parent)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as out, source.open("rb") as src:
            for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        if Path(temporary).stat().st_size != size or digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {source}, expected size={size}, SHA-256={expected}")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _object(drive, digest):
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid content SHA-256")
    path = drive / "objects" / digest[:2] / digest
    if not path.resolve().is_relative_to(drive):
        raise ValueError("Object path escapes durable storage")
    return path


def backup_area(project_root, drive_root, area: str) -> dict:
    local, drive = _paths(project_root, drive_root, area)
    if not local.is_dir():
        raise FileNotFoundError(local)
    files, signatures, transferred = [], {}, 0
    for path in _files(local):
        relative = path.relative_to(local).as_posix()
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not target.is_relative_to(local) or not target.is_file():
                raise ValueError(f"Symlink escapes cache or is not a file: {path}")
            files.append({"path": relative, "kind": "symlink", "target": target.relative_to(local).as_posix()})
            continue
        if not path.is_file():
            raise ValueError(f"Only regular files can be backed up: {path}")
        signature = _signature(path)
        digest = sha256_file(path)
        obj = _object(drive, digest)
        if not obj.exists():
            _copy_verified(path, obj, digest, signature[0])
            transferred += signature[0]
        elif not obj.is_file() or obj.stat().st_size != signature[0]:
            raise ValueError(f"Damaged durable object: {obj}")
        signatures[path] = signature
        files.append({"path": relative, "kind": "file", "size_bytes": signature[0], "sha256": digest})
    regular = {r["path"] for r in files if r["kind"] == "file"}
    if any(r["target"] not in regular for r in files if r["kind"] == "symlink"):
        raise ValueError("Symlink target is excluded or absent from this snapshot")
    inventory = {p.relative_to(local).as_posix() for p in _files(local)}
    links_unchanged = all((local / r["path"]).is_symlink() and
                          (local / r["path"]).resolve() == local / r["target"]
                          for r in files if r["kind"] == "symlink")
    if inventory != {r["path"] for r in files} or not links_unchanged or any(not p.is_file() or _signature(p) != s for p, s in signatures.items()):
        raise RuntimeError("Source changed during backup; stop writers and retry. No snapshot committed.")
    if not files:
        return {"status": "EMPTY", "area": area, "files": 0, "transferred_bytes": 0}
    payload = {"schema_version": "1.0", "created_at_utc": utc_now(), "area": area, "files": files,
               "source_project": str(Path(project_root).resolve()), "code": git_identity(Path(project_root).resolve()),
               "consistency": "file snapshot; resume must independently verify experiment provenance"}
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8] + ".json"
    manifest = atomic_json(drive / "snapshots" / area / name, {"payload": payload, "sha256": config_hash(payload)})
    return {"status": "BACKED_UP", "area": area, "files": len(files), "transferred_bytes": transferred,
            "snapshot": str(manifest), "note": "Existing objects are hash-verified on restore"}


def restore_area(project_root, drive_root, area: str, *, snapshot: str | Path | None = None) -> dict:
    local, drive = _paths(project_root, drive_root, area)
    candidates = sorted((drive / "snapshots" / area).glob("*.json"))
    if snapshot is None and not candidates:
        return {"status": "NOT_FOUND", "area": area, "files": 0}
    manifest = Path(snapshot) if snapshot is not None else candidates[-1]
    record = read_json(manifest)
    payload = record["payload"]
    if config_hash(payload) != record["sha256"] or payload.get("schema_version") != "1.0" or payload.get("area") != area:
        raise ValueError("Snapshot manifest checksum/schema/area mismatch")
    entries = payload["files"]
    by_path = {r["path"]: r for r in entries}
    if len(by_path) != len(entries):
        raise ValueError("Duplicate snapshot paths")
    # Validate the entire plan and local conflicts before publishing any files.
    for rec in entries:
        relative = _relative(rec["path"])
        if _excluded(Path(relative)):
            raise ValueError("Snapshot contains an excluded credential/temporary path")
        dest = local / relative
        if not dest.resolve().is_relative_to(local) or (dest.is_symlink() and rec["kind"] != "symlink"):
            raise ValueError(f"Restore destination is a symlink or escapes local storage: {dest}")
        if any(p.is_symlink() for p in dest.parents if p.is_relative_to(local)):
            raise ValueError(f"Restore parent must not be a symlink: {dest}")
        if rec["kind"] == "file":
            _object(drive, rec["sha256"])
            if not isinstance(rec["size_bytes"], int) or rec["size_bytes"] < 0:
                raise ValueError("Invalid snapshot size")
            if dest.exists() and (not dest.is_file() or sha256_file(dest) != rec["sha256"]):
                raise FileExistsError(f"Local file differs; restore into a fresh checkout: {dest}")
        elif rec["kind"] == "symlink":
            _relative(rec["target"])
            if rec["target"] not in by_path or by_path[rec["target"]]["kind"] != "file":
                raise ValueError("Snapshot link target must be a regular file in the same snapshot")
            if dest.is_symlink() and dest.resolve() != local / rec["target"]:
                raise ValueError(f"Local symlink has a different target: {dest}")
            if dest.exists():
                target = by_path[rec["target"]]
                if not dest.is_file() or sha256_file(dest) != target["sha256"]:
                    raise FileExistsError(f"Local link destination differs: {dest}")
        else:
            raise ValueError("Unknown snapshot file type")
    for rec in entries:
        if rec["kind"] != "file":
            continue
        dest = local / rec["path"]
        obj = _object(drive, rec["sha256"])
        # Even if a correct local file exists, check the durable copy before
        # claiming this snapshot was restored/verified.
        if dest.exists():
            if not obj.is_file() or obj.stat().st_size != rec["size_bytes"] or sha256_file(obj) != rec["sha256"]:
                raise ValueError(f"Checksum mismatch: {obj}")
        else:
            _copy_verified(obj, dest, rec["sha256"], rec["size_bytes"])
    for rec in entries:
        if rec["kind"] != "symlink":
            continue
        dest, target = local / rec["path"], local / rec["target"]
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if os.name == "posix":
                dest.symlink_to(os.path.relpath(target, dest.parent))
            else:
                # Windows often cannot create symlinks without elevated rights.
                meta = by_path[rec["target"]]
                _copy_verified(target, dest, meta["sha256"], meta["size_bytes"])
    return {"status": "RESTORED", "area": area, "files": len(entries), "snapshot": str(manifest),
            "note": "No gate or resume eligibility inferred from storage integrity"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "backup", "restore", "status"])
    parser.add_argument("--project-root", default=os.environ.get("CELLQUANT_PROJECT_ROOT", "."))
    parser.add_argument("--drive-root", default=os.environ.get("CELLQUANT_DRIVE_ROOT"))
    parser.add_argument("--areas", nargs="+", choices=list(AREAS), default=["runs"])
    parser.add_argument("--snapshot", help="Restore one explicitly selected older snapshot (one area only)")
    args = parser.parse_args()
    if not args.drive_root:
        saved = Path(args.project_root) / ".cache/colab_workspace.json"
        if saved.is_file():
            args.drive_root = read_json(saved)["environment"]["CELLQUANT_DRIVE_ROOT"]
    if not args.drive_root:
        parser.error("Set --drive-root or CELLQUANT_DRIVE_ROOT to your mounted durable directory")
    if args.snapshot and (args.action != "restore" or len(args.areas) != 1):
        parser.error("--snapshot requires restore and exactly one --areas value")
    if args.action == "init":
        result = configure_workspace(args.project_root, args.drive_root, apply=False)
    elif args.action == "status":
        result = {"code": git_identity(Path(args.project_root).resolve()), "areas": {}}
        for area in args.areas:
            local, drive = _paths(args.project_root, args.drive_root, area)
            result["areas"][area] = {"local": str(local), "snapshots": sorted(str(p) for p in (drive / "snapshots" / area).glob("*.json"))}
    else:
        fn = backup_area if args.action == "backup" else restore_area
        kw = {"snapshot": args.snapshot} if args.action == "restore" else {}
        result = [fn(args.project_root, args.drive_root, area, **kw) for area in args.areas]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
