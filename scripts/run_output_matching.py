"""Prepare fresh WikiText train documents, run the A100 matching study and archive its measurements."""
import argparse
import os
from pathlib import Path
import sys
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cellquant.runtime import config_hash, read_json, sha256_file, stream_command


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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--validate-batches', action='store_true', help='Evaluate completed batch fits on independent new documents')
    modes.add_argument('--retest-batches', action='store_true', help='Confirm the batch algorithm on fresh calibration and evaluation articles for G1')
    parser.add_argument('--all-controls', action='store_true', help='Also measure every matched solver if the parent alone rules out both G1 routes')
    parser.add_argument('--pilot-archive', type=Path, default=Path('evidence/output_matching_20261119/batch_documents_7b.zip'))
    parser.add_argument('--screen-run', type=Path, help='Completed fit screen directory; by default find the latest matching run')
    parser.add_argument('--matching-archive', type=Path, default=Path('evidence/output_matching_20261119/measurements.zip'),
                        help='01d measurement archive used in the batch screen')
    args = parser.parse_args(argv)
    parent = ROOT/args.config
    if not parent.is_file():
        parser.error(f'Original frozen G1 inputs missing at {parent}. Restore the prepared run or use --config.')
    os.environ['PYTHONPATH'] = os.pathsep.join(filter(None,(str(ROOT/'src'),os.environ.get('PYTHONPATH'))))
    if args.screen_run and not args.validate_batches:
        parser.error('--screen-run is used with --validate-batches')
    if args.all_controls and not args.retest_batches:
        parser.error('--all-controls is used with --retest-batches')
    mode = ['--matching-study']
    prepare_mode = mode
    prefix, archive_name = 'matching_', 'output_matching.zip'
    if args.retest_batches:
        mode = ['--batch-retest'] + (['--all-controls'] if args.all_controls else [])
        prepare_mode = ['--retest-pilot', str(ROOT/args.pilot_archive)]
        prefix, archive_name = 'batch_retest_', 'batch_g1_retest.zip'
    elif args.validate_batches:
        from cellquant.batch_validation import locate_fit_screen
        archive = ROOT/args.matching_archive
        screen = ROOT/args.screen_run if args.screen_run else locate_fit_screen(
            ROOT/'runs/experiments', config_hash(read_json(parent)), sha256_file(archive))
        print('Using fit screen', screen, flush=True)
        mode = ['--batch-documents']
        prepare_mode = ['--batch-document-screen', str(screen), '--matching-archive', str(archive)]
        prefix, archive_name = 'batch_documents_', 'batch_document_pilot.zip'
    work = ROOT/'runs/01_pilot'/(prefix + uuid4().hex[:12])
    inputs = work/'inputs'
    index = work/'index.json'
    print('Study directory', work, flush=True)
    commands = [('prepare', [str(ROOT/'scripts/prepare_experiment.py'),'--config',str(parent),'--output',str(inputs),*prepare_mode]),
                ('study', ['-m','cellquant.experiment','--config',str(inputs/'config.json'),*mode,'--result-index',str(index)])]
    for label, command in commands:
        status = stream_command([sys.executable,'-u',*command],work/(label+'.log'),cwd=ROOT,timeout_s=4*3600)
        if status['returncode']:
            print('Study stopped. See', status['log_path'], file=sys.stderr)
            if not (args.retest_batches and label=='study' and index.is_file()):
                return status['returncode']
    run = Path(read_json(index)['run_dir'])
    result = read_json(run/'result.json')
    if not args.retest_batches and (result['status'] != 'STUDY_COMPLETE' or result.get('source_type') != 'real_layer'):
        raise RuntimeError(f'Expected completed real measurements in {run}')
    dest = work/archive_name
    with ZipFile(dest.with_suffix('.partial'), 'w', ZIP_DEFLATED) as z:
        for path in sorted(inputs.rglob('*')):
            if path.is_file():
                z.write(path, Path('inputs')/path.relative_to(inputs))
        for path in sorted(run.rglob('*.json')):
            z.write(path, Path('run')/path.relative_to(run))
    dest.with_suffix('.partial').replace(dest)
    if args.retest_batches:
        print('Formal batch retest', result['status'], 'G1', result.get('gate','not evaluated'))
        if 'route_bounds' in result:
            print('Upper bounds', result['route_bounds'])
        if result.get('reason'):
            print('Reason', result['reason'])
    elif args.validate_batches:
        stats = result['document_summary']
        print('Fresh paired documents', stats['distinct_documents'])
        for label, item in stats['comparisons'].items():
            if 'undefined' in item:
                print(label, item['undefined'])
                continue
            effect = item['all']
            lo, hi = effect['paired_95_pct']
            print(f"{label}: all sites {effect['median_gain_pct']:+.4f}% [{lo:+.4f}, {hi:+.4f}], seed medians {effect['seed_medians_pct']}")
        print('Worst site against grid', stats.get('worst_site_vs_grid'))
        for seed, times in result['calibration_seconds_by_seed'].items():
            print(f"Seed {seed}: grid {times['continued_grid_seconds']:.2f} s, "
                  f"grid + proposal + fit screen {times['accounted_candidate_seconds']:.2f} s, "
                  f"random control within fit budgets {times['random_fit_feasible']}")
        print('next_step', result['next_step'])
    else:
        print_results(result['matching_summary'])
    print('Results', dest)
    return status['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
