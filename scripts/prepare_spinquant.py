"""Materialize a hashed compatibility overlay, preserving the pinned checkout."""
import argparse
from pathlib import Path
from cellquant.spinquant_adapter import prepare_overlay

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='third_party/SpinQuant')
    p.add_argument('--output',default='runs/spinquant_overlay_v2');p.add_argument('--fp32-port',action='store_true')
    args=p.parse_args()
    print(prepare_overlay(Path(args.source),Path(args.output),fp32_port=args.fp32_port))
