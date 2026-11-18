"""Prepare fresh WikiText train documents, run the A100 matching study and archive its measurements."""
import argparse
import os
from pathlib import Path
import sys
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cellquant.runtime import read_json, stream_command


def print_results(summary):
    print(f"Paired held-out documents: {summary['distinct_documents']}")
    for label, group in (('primary_vs_warm_grid','output'), ('activation_information','output'),
                         ('coverage','output'), ('hybrid_vs_continued_grid','output'),
                         ('hybrid_vs_continued_grid','input'), ('hybrid_vs_continued_grid','all')):
        comparison = summary['comparisons'][label]
        if 'undefined' in comparison:
            print(f'{label} / {group}: {comparison["undefined"]}')
            continue
        effect = comparison[group]
        lo, hi = effect['paired_95_pct']
        seeds = ', '.join(f'{s}={v:+.2f}%' for s,v in effect['seed_medians_pct'].items())
        print(f"{label} / {group}: {effect['median_gain_pct']:+.2f}% [{lo:+.2f}, {hi:+.2f}], seeds {seeds}")
    for seed, row in summary['hybrid_mechanism_by_seed'].items():
        times = row['calibration_seconds']
        extra = times['continued_grid_plus_matching'] - times['continued_grid']
        print(f"Hybrid seed {seed}: applied {row['applied_pairs']} pairs, additional calibration {extra:.2f} s")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=os.environ.get('CELLQUANT_PILOT_CONFIG','runs/prepared/g1_gpu/config.json'),
                        help='Pinned original G1 input config in the current checkout')
    args = parser.parse_args(argv)
    parent = ROOT/args.config
    if not parent.is_file():
        parser.error(f'Original frozen G1 inputs missing at {parent}. Restore the prepared run or use --config.')
    os.environ['PYTHONPATH'] = os.pathsep.join(filter(None,(str(ROOT/'src'),os.environ.get('PYTHONPATH'))))
    work = ROOT/'runs/01_pilot'/('matching_' + uuid4().hex[:12])
    inputs = work/'inputs'
    index = work/'index.json'
    print('Study directory', work, flush=True)
    commands = [('prepare', [str(ROOT/'scripts/prepare_experiment.py'),'--config',str(parent),'--output',str(inputs),'--matching-study']),
                ('study', ['-m','cellquant.experiment','--config',str(inputs/'config.json'),'--matching-study','--result-index',str(index)])]
    for label, command in commands:
        status = stream_command([sys.executable,'-u',*command],work/(label+'.log'),cwd=ROOT,timeout_s=4*3600)
        if status['returncode']:
            print('Study stopped. See', status['log_path'], file=sys.stderr)
            return status['returncode']
    run = Path(read_json(index)['run_dir'])
    result = read_json(run/'result.json')
    if result['status'] != 'STUDY_COMPLETE' or result.get('source_type') != 'real_layer':
        raise RuntimeError(f'Expected completed real measurements in {run}')
    dest = work/'output_matching.zip'
    with ZipFile(dest.with_suffix('.partial'), 'w', ZIP_DEFLATED) as z:
        for path in sorted(inputs.rglob('*')):
            if path.is_file():
                z.write(path, Path('inputs')/path.relative_to(inputs))
        for path in sorted(run.rglob('*.json')):
            z.write(path, Path('run')/path.relative_to(run))
    dest.with_suffix('.partial').replace(dest)
    print_results(result['matching_summary'])
    print('Results', dest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
