"""Live execution telemetry. Heartbeats are liveness, never scientific evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

import psutil

from .runtime import atomic_json, utc_now


class RunProgress:
    def __init__(self, output, *, interval=30.):
        if interval <= 0:
            raise ValueError('Positive heartbeat interval required')
        self.output = Path(output)
        self.interval = interval
        self.process = psutil.Process()
        self.started = self.phase_started = time.monotonic()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.error = None
        self.state = {
            'run_id': self.output.name, 'run_dir': str(self.output.resolve()),
            'pid': os.getpid(), 'process_create_time': self.process.create_time(),
            'status': 'RUNNING', 'phase': 'initialization', 'details': {},
            'last_progress_at_utc': utc_now(), 'progress_sequence': 0,
            'scope': 'Execution telemetry only; a heartbeat does not mean computation advanced',
        }
        self.step('initialization')
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _write(self, event):
        memory = psutil.virtual_memory()
        cpu = self.process.cpu_times()
        now = time.monotonic()
        self.state.update(
            event=event, updated_at_utc=utc_now(), elapsed_seconds=round(now-self.started, 2),
            phase_elapsed_seconds=round(now-self.phase_started, 2),
            process_rss_gib=round(self.process.memory_info().rss/2**30, 3),
            process_cpu_seconds=round(cpu.user+cpu.system, 2),
            system_available_gib=round(memory.available/2**30, 3),
            system_total_gib=round(memory.total/2**30, 3),
        )
        for attempt in range(6):
            try:
                atomic_json(self.output/'state.json', self.state)
                break
            except PermissionError:
                # Windows readers briefly hold non-delete-sharing handles.
                # Retry only this bounded sharing race; persistent I/O errors propagate.
                if os.name != 'nt' or attempt == 5:
                    raise
                time.sleep(.01*(attempt+1))
        fields = ['event', 'status', 'phase', 'details', 'elapsed_seconds',
                  'phase_elapsed_seconds', 'process_rss_gib', 'system_available_gib',
                  'process_cpu_seconds', 'progress_sequence', 'updated_at_utc']
        print('[CellQuant] '+json.dumps({k: self.state[k] for k in fields}), flush=True)

    def step(self, phase, **details):
        with self.lock:
            if self.error is not None:
                raise RuntimeError('Progress heartbeat failed') from self.error
            if self.state['status'] != 'RUNNING':
                raise RuntimeError('Cannot advance a closed run')
            if phase != self.state['phase']:
                self.phase_started = time.monotonic()
            self.state.update(phase=phase, details=details, last_progress_at_utc=utc_now())
            self.state['progress_sequence'] += 1
            self._write('progress')

    def _heartbeat(self):
        try:
            while not self.stop_event.wait(self.interval):
                with self.lock:
                    self._write('heartbeat')
        except Exception as exc:
            # Surface background I/O failures on the next foreground update/close.
            self.error = exc
            print('[CellQuant] progress heartbeat failed: '+type(exc).__name__, flush=True)

    def close(self, status):
        self.stop_event.set()
        self.thread.join()
        with self.lock:
            if self.error is not None:
                raise RuntimeError('Progress heartbeat failed') from self.error
            self.state['status'] = status
            self._write('finished')


def report(progress, phase, **details):
    if progress is not None:
        progress(phase, **details)
