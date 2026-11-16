"""Execute storage/Notebook fixtures; this does not contact Drive or run a model."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import nbformat

from cellquant.runtime import atomic_json, sha256_file
from cellquant.workspace import backup_area, configure_workspace, git_identity


def main():
    root = Path(__file__).resolve().parents[2]
    evidence = Path(__file__).resolve().parent
    notebook = root / "notebooks/00_Colab_GitHub_Workspace.ipynb"
    nb = nbformat.read(notebook, as_version=4)
    nbformat.validate(nb)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook-cell-{index}", "exec")
    saved = nbformat.read(evidence / "user_workspace_before.ipynb", as_version=4)
    old_config = saved.cells[1].source
    new_config = nb.cells[1].source
    # Only the new conflict option may change the user's configuration.
    assert "\n".join(l for l in new_config.splitlines() if not l.startswith("RESTORE_RUNS_CONFLICT")) == old_config.rstrip("\n")
    unchanged = 0
    for old in saved.cells:
        if old.cell_type == "code" and old.source not in (old_config,) and "def logged_command" not in old.source:
            new = next(c for c in nb.cells if c.get("id") == old.get("id"))
            assert new == old
            unchanged += 1

    fixture = Path(tempfile.mkdtemp(prefix="restore_cli_", dir=root / "runs"))
    source, target, drive = (fixture / name for name in ("source", "target", "drive"))
    for checkout in (source, target):
        checkout.mkdir()
        (checkout / "pyproject.toml").write_text("[project]\nname='fixture'\n")
        configure_workspace(checkout, drive, apply=False)
        (checkout / "runs/00_math").mkdir()
    original = target / "runs/00_math/environment.json"
    original.write_bytes(b"current environment fixture")
    (source / "runs/00_math/environment.json").write_bytes(b"old environment fixture")
    backup_area(source, drive, "runs")
    command = [sys.executable, "-m", "cellquant.workspace", "restore", "--project-root", str(target),
               "--drive-root", str(drive), "--areas", "runs"]
    rejected = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert rejected.returncode != 0 and "refusing to overwrite" in rejected.stderr
    (evidence / "cli_default_conflict.log").write_text(rejected.stdout + rejected.stderr, encoding="utf-8")

    os.environ["CELLQUANT_PROJECT_ROOT"] = str(target)
    os.environ["CELLQUANT_DRIVE_ROOT"] = str(drive)
    env = {"PROJECT_ROOT": target, "sys": sys, "INSTALL_BASE_DEPENDENCIES": False,
           "RESTORE_AREAS": ["runs"], "RESTORE_RUNS_CONFLICT": "separate", "WORKSPACE_READY": True}
    helper_cell = next(c.source for c in nb.cells if c.cell_type == "code" and "def logged_command" in c.source)
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(helper_cell, "actual-notebook-install-restore-cell", "exec"), env)
    (evidence / "notebook_restore.log").write_text(output.getvalue(), encoding="utf-8")
    log = next((target / ".cache/workspace_logs").glob("restore_*.log"))
    result = json.loads(log.read_text(encoding="utf-8"))[0]
    assert result["status"] == "RESTORED_SEPARATELY"
    assert original.read_bytes() == b"current environment fixture"
    separate = Path(result["destination"])
    assert not separate.is_relative_to(target / "runs")
    assert (separate / "00_math/environment.json").read_bytes() == b"old environment fixture"

    with redirect_stdout(output):
        try:
            env["logged_command"]([sys.executable, "-c", "import sys; print('fixture diagnostic detail'); sys.exit(7)"],
                                  "expected_failure", 30)
        except RuntimeError as exc:
            assert "fixture diagnostic detail" in str(exc) and "exit 7" in str(exc)
            (evidence / "notebook_expected_failure.log").write_text(str(exc), encoding="utf-8")
        else:
            raise AssertionError("Notebook swallowed a failed command")

    # The option applies only to runs when restoring several areas together.
    multi = subprocess.run(command[:-1] + ["models", "datasets", "runs", "--runs-conflict", "separate"],
                           capture_output=True, text=True, encoding="utf-8", timeout=30)
    (evidence / "cli_multi_area.log").write_text(multi.stdout + multi.stderr, encoding="utf-8")
    assert multi.returncode == 0, multi.stderr
    assert [r["status"] for r in json.loads(multi.stdout)] == ["NOT_FOUND", "NOT_FOUND", "RESTORED_SEPARATELY"]

    report = {
        "source_type": "unit_test_fixture", "status": "PASS", "fixture_directory": str(fixture),
        "python": sys.version, "code": git_identity(root),
        "checks": ["notebook schema and code syntax", "user configuration and unchanged outputs preserved",
                   "CLI default conflict rejection", "actual notebook separate restore call",
                   "both original and historical bytes intact", "error contains child diagnostic and exit code",
                   "multi-area option only applies to runs"],
        "unchanged_code_cells": unchanged,
        "source_sha256": {str(p.relative_to(root)): sha256_file(p) for p in
                          (notebook, root / "src/cellquant/workspace.py", root / "tests/test_workspace.py")},
        "user_notebook_sha256": sha256_file(evidence / "user_workspace_before.ipynb"),
        "gate_rules_sha256": sha256_file(root / "configs/gate_rules.json"),
        "colab_drive_retest": "NOT_RUN", "real_model_gpu": "NOT_RUN",
    }
    assert report["gate_rules_sha256"] == "5d36290ca487795337300f3cccbc1c4d5a18d250639498bb15455ca5ad6a554e"
    atomic_json(evidence / "validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
