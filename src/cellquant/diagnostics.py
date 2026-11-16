"""Read-only local process/run inspection; does not import torch or load a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import psutil


def _read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return {'read_error': type(exc).__name__}


def inspect_runs(project_root, *, sample_seconds=1.):
    """A live PID is checked against its command and cwd, never trusted from a stale file."""
    root = Path(project_root).resolve()
    processes = []
    errors = []
    for process in psutil.process_iter():
        try:
            argv = process.cmdline()
            if not any(argv[i:i+2] == ['-m', 'cellquant.experiment'] for i in range(len(argv)-1)):
                continue
            if Path(process.cwd()).resolve() != root:
                continue
            process.cpu_percent(None)
            processes.append(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied:
            errors.append({'pid': process.pid, 'read_error': 'AccessDenied'})
    if processes:
        time.sleep(sample_seconds)
    live = []
    for process in processes:
        try:
            cpu = process.cpu_times()
            live.append({'pid': process.pid, 'process_create_time': process.create_time(),
                         'status': process.status(), 'cpu_percent': process.cpu_percent(None),
                         'cpu_seconds': cpu.user+cpu.system,
                         'rss_gib': process.memory_info().rss/2**30,
                         'elapsed_seconds': time.time()-process.create_time()})
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            live.append({'pid': process.pid, 'status': 'EXITED_DURING_SAMPLE'})
        except psutil.AccessDenied:
            live.append({'pid': process.pid, 'read_error': 'AccessDenied'})
    runs = []
    paths = list((root/'runs/experiments').glob('*/state.json'))
    for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
        state = _read(path)
        # Whitelist telemetry fields; never return configs, argv or credentials.
        row = {k: state[k] for k in ['status', 'phase', 'details', 'pid', 'updated_at_utc',
                    'last_progress_at_utc', 'progress_sequence', 'event', 'read_error'] if k in state}
        row['run_dir'] = str(path.parent)
        row['state_age_seconds'] = time.time()-path.stat().st_mtime
        matches = [p for p in live if p.get('pid') == state.get('pid') and
                   abs(p.get('process_create_time', 0)-state.get('process_create_time', -1)) < .1]
        row['live_process_verified'] = bool(matches)
        if 'progress_sequence' not in state:
            row['note'] = 'Legacy state: phase may remain fusion during capture/solve; cannot identify live phase'
        result_path = path.parent/'result.json'
        if result_path.is_file():
            row['result_status'] = _read(result_path).get('status')
        row['completed_method_artifacts'] = len(list(path.parent.glob('seed_*/*/pilot_result.json')))
        runs.append(row)
    memory = psutil.virtual_memory()
    return {'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'project_root': str(root),
            'processes': live, 'recent_runs': runs,
            'system_memory': {'available_gib': memory.available/2**30, 'total_gib': memory.total/2**30,
                              'swap_used_gib': psutil.swap_memory().used/2**30},
            'process_read_errors': errors,
            'scope': 'One resource sample cannot prove progress or deadlock. Compare CPU time/progress across samples; no gate evidence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    args = parser.parse_args()
    print(json.dumps(inspect_runs(args.project_root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
