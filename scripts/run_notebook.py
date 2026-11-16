#!/usr/bin/env python3
"""Execute a notebook with nbclient, preserving an executed copy and exit status."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import os
import signal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800, help="Per-cell seconds")
    parser.add_argument("--engine", choices=["nbclient", "inprocess"], default="nbclient",
                        help="inprocess runs IPython cells without a Jupyter socket; records the distinction")
    args = parser.parse_args()
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(args.notebook, as_version=4)
    nbformat.validate(nb)
    root = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nb.metadata.setdefault("cellquant", {})["validation_engine"] = args.engine
    try:
        if args.engine == "nbclient":
            client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                                    resources={"metadata": {"path": str(root)}},
                                    allow_errors=False)
            client.execute()
        else:
            from IPython.core.interactiveshell import InteractiveShell
            from IPython.utils.capture import capture_output
            shell = InteractiveShell.instance()
            shell.user_ns["__name__"] = "__main__"
            previous_cwd = Path.cwd()
            os.chdir(root)
            count = 0
            def timeout_handler(signum, frame):
                raise TimeoutError("Notebook cell exceeded the configured timeout")
            prior_handler = signal.signal(signal.SIGALRM, timeout_handler) if hasattr(signal, "SIGALRM") else None
            try:
                for cell in nb.cells:
                    if cell.cell_type != "code":
                        continue
                    count += 1
                    cell.execution_count = count
                    cell.outputs = []
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(args.timeout)
                    try:
                        with capture_output() as captured:
                            result = shell.run_cell(cell.source, store_history=False)
                    finally:
                        if hasattr(signal, "SIGALRM"):
                            signal.alarm(0)
                    if captured.stdout:
                        cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
                    if captured.stderr:
                        cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
                    for output in captured.outputs:
                        cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
                    if result.error_before_exec or result.error_in_exec:
                        raise result.error_before_exec or result.error_in_exec
            finally:
                os.chdir(previous_cwd)
                if prior_handler is not None:
                    signal.signal(signal.SIGALRM, prior_handler)
    finally:
        nbformat.write(nb, args.output)
    print(f"Executed notebook saved to {args.output}")

if __name__ == "__main__":
    main()
