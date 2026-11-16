#!/usr/bin/env python3
"""Run CPU math checks; a PASS is not a full-model research result."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cellquant.math_reference import run_reference_checks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'math_reference')
    parser.add_argument('--seed', type=int, default=20261115)
    args = parser.parse_args()
    report = run_reference_checks(args.output, args.seed)
    print(json.dumps({'status': report['status'], 'checks':report['checks'], 'report':str(args.output / 'math_reference_report.json')}, indent=2))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)
