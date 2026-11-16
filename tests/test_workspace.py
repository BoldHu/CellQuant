"""Storage integrity/control-flow fixtures, never model or gate evidence."""
import json
from pathlib import Path

import pytest

from cellquant.runtime import atomic_json, config_hash, huggingface_cache_paths
from cellquant.workspace import (AREAS, backup_area, configure_workspace, restore_area,
                                 workspace_environment)


def checkout(path):
    path.mkdir()
    (path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    return path


def test_roundtrip_deduplicates_and_preserves_versions_without_credentials(tmp_path):
    source, restored = checkout(tmp_path / "source"), checkout(tmp_path / "restored")
    drive = tmp_path / "drive"
    configure_workspace(source, drive, apply=False)
    configure_workspace(restored, drive, apply=False)
    cache = source / AREAS["models"]
    (cache / "weight.safetensors").write_bytes(b"fixture bytes, not a real model")
    (cache / "token").write_text("must not be backed up")
    (cache / "download.incomplete").write_text("partial")
    (cache.parent / "stored_tokens").write_text("also not backed up")
    first = backup_area(source, drive, "models")
    second = backup_area(source, drive, "models")
    assert first["transferred_bytes"] > 0 and second["transferred_bytes"] == 0
    assert first["snapshot"] != second["snapshot"]
    assert restore_area(restored, drive, "models")["status"] == "RESTORED"
    assert (restored / AREAS["models"] / "weight.safetensors").read_bytes() == (cache / "weight.safetensors").read_bytes()
    assert not (restored / AREAS["models"] / "token").exists()
    assert not (restored / AREAS["models"] / "download.incomplete").exists()
    assert restore_area(restored, drive, "models")["status"] == "RESTORED"
    (cache / "weight.safetensors").write_bytes(b"new revision")
    backup_area(source, drive, "models")
    with pytest.raises(FileExistsError, match="differs"):
        restore_area(restored, drive, "models")
    assert restore_area(restored, drive, "models", snapshot=first["snapshot"])["status"] == "RESTORED"


def test_corrupted_object_and_manifest_cannot_restore(tmp_path):
    source, target = checkout(tmp_path / "source"), checkout(tmp_path / "target")
    drive = tmp_path / "drive"
    configure_workspace(source, drive, apply=False)
    (source / "runs/result.json").write_bytes(b"original")
    snapshot = Path(backup_area(source, drive, "runs")["snapshot"])
    rec = json.loads(snapshot.read_text())
    digest = rec["payload"]["files"][0]["sha256"]
    (drive / "objects" / digest[:2] / digest).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        restore_area(target, drive, "runs")
    assert not (target / "runs/result.json").exists()
    rec["payload"]["files"][0]["path"] = "../escape"
    atomic_json(snapshot, rec)
    with pytest.raises(ValueError, match="manifest checksum"):
        restore_area(target, drive, "runs")
    rec["sha256"] = config_hash(rec["payload"])
    atomic_json(snapshot, rec)
    with pytest.raises(ValueError, match="Unsafe snapshot"):
        restore_area(target, drive, "runs")


def test_runs_conflict_restores_whole_snapshot_separately_without_overwriting(tmp_path):
    source, target = checkout(tmp_path / "source"), checkout(tmp_path / "target")
    drive = tmp_path / "drive"
    configure_workspace(source, drive, apply=False)
    configure_workspace(target, drive, apply=False)
    for root, data in ((source, b"old environment"), (target, b"current environment")):
        path = root / "runs/00_math/environment.json"
        path.parent.mkdir()
        path.write_bytes(data)
    (source / "runs/00_math/result.json").write_bytes(b"old result fixture")
    (target / "runs/current.txt").write_bytes(b"new local result")
    snapshot = backup_area(source, drive, "runs")["snapshot"]
    with pytest.raises(FileExistsError, match="--runs-conflict separate"):
        restore_area(target, drive, "runs")
    assert not (target / "runs/00_math/result.json").exists()
    result = restore_area(target, drive, "runs", on_conflict="separate")
    assert result["status"] == "RESTORED_SEPARATELY" and result["snapshot"] == snapshot
    separate = Path(result["destination"])
    assert separate.is_relative_to(target / "restored_snapshots/runs")
    assert not separate.is_relative_to(target / "runs")
    assert (separate / "00_math/environment.json").read_bytes() == b"old environment"
    assert (separate / "00_math/result.json").read_bytes() == b"old result fixture"
    assert (target / "runs/00_math/environment.json").read_bytes() == b"current environment"
    assert (target / "runs/current.txt").read_bytes() == b"new local result"
    assert not (target / "runs/00_math/result.json").exists()  # No mixing of old/new runs.
    again = restore_area(target, drive, "runs", on_conflict="separate")
    assert again["destination"] == result["destination"]
    # Neither corrupt separate copies nor remote objects may be silently replaced.
    (separate / "00_math/result.json").write_bytes(b"changed historical result")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        restore_area(target, drive, "runs", on_conflict="separate")


def test_separate_runs_restore_still_checks_checksums_and_all_paths(tmp_path):
    source, target = checkout(tmp_path / "source"), checkout(tmp_path / "target")
    drive = tmp_path / "drive"
    configure_workspace(source, drive, apply=False)
    configure_workspace(target, drive, apply=False)
    (source / "runs/a.json").write_bytes(b"old")
    (target / "runs/a.json").write_bytes(b"current")
    manifest = Path(backup_area(source, drive, "runs")["snapshot"])
    record = json.loads(manifest.read_text())
    entry = record["payload"]["files"][0]
    obj = drive / "objects" / entry["sha256"][:2] / entry["sha256"]
    obj.write_bytes(b"bad")  # Same length, wrong SHA.
    with pytest.raises(ValueError, match="Checksum mismatch"):
        restore_area(target, drive, "runs", on_conflict="separate")
    assert (target / "runs/a.json").read_bytes() == b"current"
    assert not list((target / "restored_snapshots/runs").rglob("a.json"))
    obj.write_bytes(b"old")
    # The first original destination conflicts, but validation must still reach
    # later invalid entries before copying any part of the separate snapshot.
    record["payload"]["files"].append({**entry, "path": "../escape"})
    record["sha256"] = config_hash(record["payload"])
    atomic_json(manifest, record)
    with pytest.raises(ValueError, match="Unsafe snapshot"):
        restore_area(target, drive, "runs", on_conflict="separate")
    assert not list((target / "restored_snapshots/runs").rglob("a.json"))
    assert not (target / "runs/escape").exists()


def test_separate_policy_keeps_fresh_restores_in_place_and_rejects_cache_redirects(tmp_path):
    source, target = checkout(tmp_path / "source"), checkout(tmp_path / "target")
    drive = tmp_path / "drive"
    configure_workspace(source, drive, apply=False)
    (source / "runs/result.json").write_bytes(b"fixture")
    backup_area(source, drive, "runs")
    result = restore_area(target, drive, "runs", on_conflict="separate")
    assert result["status"] == "RESTORED" and Path(result["destination"]) == target / "runs"
    assert (target / "runs/result.json").read_bytes() == b"fixture"
    assert not (target / "restored_snapshots").exists()
    for area in ("models", "datasets"):
        with pytest.raises(ValueError, match="only supported for runs"):
            restore_area(target, drive, area, on_conflict="separate")


def test_interrupted_copy_never_commits_snapshot(tmp_path, monkeypatch):
    import cellquant.workspace as workspace
    source = checkout(tmp_path / "source")
    drive = tmp_path / "drive"
    configure_workspace(source, drive, apply=False)
    (source / "runs/state.npz").write_bytes(b"checkpoint fixture")
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(workspace, "_copy_verified", interrupt)
    with pytest.raises(KeyboardInterrupt):
        backup_area(source, drive, "runs")
    assert not list((drive / "snapshots").rglob("*.json"))
    assert restore_area(source, drive, "runs")["status"] == "NOT_FOUND"


def test_changing_source_cannot_publish_snapshot(tmp_path, monkeypatch):
    import cellquant.workspace as workspace
    source = checkout(tmp_path / "source")
    drive = tmp_path / "drive"
    configure_workspace(source, drive, apply=False)
    path = source / "runs/state.npz"
    path.write_bytes(b"first")
    original = workspace._copy_verified
    def change(src, *args):
        original(src, *args)
        src.write_bytes(b"changed meanwhile")
    monkeypatch.setattr(workspace, "_copy_verified", change)
    with pytest.raises(RuntimeError, match="changed during backup"):
        backup_area(source, drive, "runs")
    assert not list((drive / "snapshots").rglob("*.json"))


def test_missing_mount_and_overlapping_trees_are_rejected(tmp_path):
    source = checkout(tmp_path / "source")
    with pytest.raises(FileNotFoundError, match="parent"):
        configure_workspace(source, tmp_path / "not_mounted" / "store", apply=False)
    with pytest.raises(ValueError, match="separate trees"):
        configure_workspace(source, source / "drive", apply=False)
    assert not (source / "drive").exists()


def test_workspace_environment_and_frozen_cache_override(tmp_path, monkeypatch):
    source = checkout(tmp_path / "source")
    env = workspace_environment(source)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    models, datasets = huggingface_cache_paths()
    assert models == source / AREAS["models"]
    assert datasets == source / AREAS["datasets"]
    fixed = tmp_path / "existing_frozen_cache"
    assert huggingface_cache_paths({"cache_dir": str(fixed)})[0] == fixed
    assert all("token" not in p.parts for p in (models, datasets))


def test_symlink_manifest_roundtrip_and_outside_target_rejected(tmp_path):
    # Construct the manifest explicitly so Windows without symlink privilege
    # still executes the link reconstruction and traversal validation paths.
    source, target = checkout(tmp_path / "source"), checkout(tmp_path / "target")
    drive = tmp_path / "drive"
    configure_workspace(source, drive, apply=False)
    (source / AREAS["models"] / "blob").write_bytes(b"snapshot fixture")
    manifest = Path(backup_area(source, drive, "models")["snapshot"])
    rec = json.loads(manifest.read_text())
    rec["payload"]["files"].append({"path": "snapshots/revision/model.bin", "kind": "symlink", "target": "blob"})
    rec["sha256"] = config_hash(rec["payload"])
    atomic_json(manifest, rec)
    restore_area(target, drive, "models")
    assert (target / AREAS["models"] / "snapshots/revision/model.bin").read_bytes() == b"snapshot fixture"
    restore_area(target, drive, "models")
    rec["payload"]["files"][-1]["target"] = "../../outside"
    rec["sha256"] = config_hash(rec["payload"])
    atomic_json(manifest, rec)
    with pytest.raises(ValueError, match="Unsafe snapshot path"):
        restore_area(target, drive, "models")


def test_empty_cache_is_not_a_successful_backup(tmp_path):
    source = checkout(tmp_path / "source")
    drive = tmp_path / "drive"
    configure_workspace(source, drive, apply=False)
    assert backup_area(source, drive, "datasets")["status"] == "EMPTY"


def test_cache_model_verifies_pinned_files_without_loading_model(tmp_path, monkeypatch):
    import importlib.util
    import numpy as np
    hub = pytest.importorskip("huggingface_hub")
    safetensors = pytest.importorskip("safetensors.numpy")
    pytest.importorskip("torch")
    from cellquant.runtime import sha256_file
    spec = importlib.util.spec_from_file_location("cache_model_script", Path(__file__).resolve().parents[1] / "scripts/cache_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cached = tmp_path / "cached"
    cached.mkdir()
    weight = cached / "model.safetensors"
    safetensors.save_file({"fixture": np.zeros((2, 3), dtype=np.float32)}, weight)
    source = atomic_json(tmp_path / "source.json", {"model_id": "fixture/tiny", "revision": "a" * 40,
        "files": [{"path": weight.name, "size_bytes": weight.stat().st_size, "sha256": sha256_file(weight)}]})
    cfg = {"model_id": "fixture/tiny", "model_revision": "a" * 40, "tokenizer_revision": "b" * 40,
           "model_source_manifest_path": str(source), "model_source_manifest_sha256": sha256_file(source),
           "runtime": {"cache_dir": str(tmp_path / "cache")}}
    config = atomic_json(tmp_path / "config.json", cfg)
    calls = []
    def download(repo, **kw):
        calls.append((repo, kw["revision"], kw["cache_dir"]))
        return str(cached)
    monkeypatch.setattr(hub, "snapshot_download", download)
    result = module.cache_model(config, tmp_path / "verification")
    assert result["status"] == "WEIGHTS_VERIFIED" and result["source_type"] == "not_run"
    assert {c[1] for c in calls} == {"a" * 40, "b" * 40}
    assert all(c[2] == str(tmp_path / "cache") for c in calls)
    weight.write_bytes(b"corrupt")
    from cellquant.experiment import ExperimentBlocked
    with pytest.raises(ExperimentBlocked, match="Corrupt weight"):
        module.cache_model(config, tmp_path / "verification")
    calls.clear()
    cfg["model_revision"] = "main"
    atomic_json(config, cfg)
    with pytest.raises(ValueError, match="immutable"):
        module.cache_model(config, tmp_path / "verification")
    assert not calls
