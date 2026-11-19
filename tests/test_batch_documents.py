import copy
import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip('torch')
from cellquant import batch_validation as study
from cellquant.matching_batches import load_matching_arms
from cellquant.matching_study import CONTINUED
from cellquant.runtime import atomic_json, config_hash, read_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'evidence/output_matching_20261118/hybrid_42m/measurements.zip'
SEEDS = [11, 23]


def fit_receipt(folder, parent):
    # Original 42M measurements; supply the two outer run-index fields used by the Colab runner.
    fit = read_json(ROOT/'evidence/output_matching_20261119/fit_batches_42m.json')
    for seed, row in fit.pop('by_seed').items():
        path = atomic_json(folder/f'seed_{seed}'/'batch_fit.json', row)
        assert sha256_file(path) == fit['fit_screen'][seed]['fit_trace_sha256']
    return atomic_json(folder/'result.json', {**fit, 'run_id': folder.name, 'config_sha256': parent,
                                             'created_at_utc': '2026-11-19T08:00:00+00:00'})


def test_frozen_batch_receipts_and_colab_entry(tmp_path, monkeypatch, capsys):
    parent = atomic_json(tmp_path/'prepared.json', {'stage': 'G1'})
    digest = config_hash(read_json(parent))
    experiments = tmp_path/'runs/experiments'
    good = experiments/'fit'
    receipt = fit_receipt(good, digest)
    fit_receipt(experiments/'other', 'other-parent')
    assert study.locate_fit_screen(experiments, digest, sha256_file(ARCHIVE)) == good
    spec = importlib.util.spec_from_file_location('batch_colab_cli', ROOT/'scripts/run_output_matching.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setenv('PYTHONPATH', '')
    calls = []

    def execute(argv, log, cwd, timeout_s):
        calls.append(Path(log).stem)
        if '--output' in argv:
            assert argv[argv.index('--batch-document-screen')+1] == str(good)
            inputs = Path(argv[argv.index('--output')+1])
            inputs.mkdir(parents=True)
            source = study.freeze_batch_inputs(inputs, good, ARCHIVE, SEEDS, digest)
            assert source['screen_result_sha256'] == sha256_file(receipt)
        else:
            assert '--batch-documents' in argv and '--matching-study' not in argv
            run = tmp_path/'score'
            atomic_json(run/'result.json', {'status': 'STUDY_COMPLETE', 'source_type': 'real_layer',
                        'document_summary': {'distinct_documents': 32, 'comparisons': {}},
                        'calibration_seconds_by_seed': {}, 'next_step': 'stop_batch_variant'})
            atomic_json(Path(argv[argv.index('--result-index')+1]), {'run_dir': str(run)})
        return {'returncode': 0, 'log_path': str(log)}

    monkeypatch.setattr(cli, 'stream_command', execute)
    assert cli.main(['--config', str(parent), '--matching-archive', str(ARCHIVE), '--validate-batches']) == 0
    assert calls == ['prepare', 'study']
    assert 'stop_batch_variant' in capsys.readouterr().out
    package, = (tmp_path/'runs/01_pilot').glob('batch_documents_*/batch_document_pilot.zip')
    with ZipFile(package) as z:
        assert 'inputs/fit_reference/seed_23/batch_fit.json' in z.namelist()
        assert 'run/result.json' in z.namelist()
    trace = good/'seed_23/batch_fit.json'
    trace.write_bytes(trace.read_bytes()+b' ')
    with pytest.raises(ValueError, match='checksum'):
        study.verified_screen(good, SEEDS, sha256_file(ARCHIVE), digest)


def test_fresh_cohort_excludes_measured_and_reserved_documents_in_both_seeds():
    def block(i):
        return {'document_id': str(i), 'block_id': f'{i}:0', 'document_sha256': f'doc{i}', 'token_ids_sha256': f'tokens{i}'}
    blocks = [block(i) for i in range(96)]
    previous = {s: {'fit': blocks[i:i+4], 'dev': blocks[i+4:i+8]} for s,i in zip(SEEDS, (0,8))}
    arms = {s: {CONTINUED: {'fit_documents': [b['document_id'] for b in previous[s]['fit']],
                 'heldout': [{'blocks': blocks[i:i+8]}]}} for s,i in zip(SEEDS, (16,24))}
    reserved = blocks[32:48]
    pool = blocks + [{**block(98), 'document_sha256': 'doc29'}, {**block(99), 'token_ids_sha256': 'tokens19'}]
    fresh, excluded = study.draw_fresh_documents(pool, previous, reserved, arms)
    assert fresh == study.draw_fresh_documents(list(reversed(pool)), previous, reserved, arms)[0]
    assert all(p[0] == previous[s]['fit'] and len(p[1]) == 16 for s,p in fresh.items())
    unused = [b for _,dev in fresh.values() for b in dev]
    for key in ('document_id', 'document_sha256', 'token_ids_sha256'):
        observed = {b[key] for b in unused}
        assert len(observed) == 32 and observed.isdisjoint(b[key] for group in excluded.values() for b in group)
    with pytest.raises(ValueError, match='actual 01d'):
        study.draw_fresh_documents(blocks[:31], previous, reserved, arms)


def test_batch_decision_checks_both_fits_and_document_uncertainty():
    sites = {f'site{i}': ('input' if i < 6 else 'output') for i in range(9)}
    methods = (CONTINUED, study.SELECTED, study.RANDOM)
    def rows(ratios, grid=10., *, variable=False):
        def squared_error(seed, method, i):
            if method == CONTINUED:
                return grid
            if method == study.RANDOM:
                return 1.
            noise = (6. if i < 8 else -6.) if variable else 0.
            return ratios[seed] + noise
        return [{'seed': s, 'method': m, 'panels': {
            site: [{'document_id': f'{s}_{i}', 'block_id': f'{s}_{i}:0', 'target_energy': 100., 'sampled_tokens': 32,
                    'squared_error': squared_error(s, m, i)}
                   for i in range(16)] for site in sites}} for s in SEEDS for m in methods]
    stats, decision = study.assess_documents(rows({11: 8., 23: 9.}), SEEDS, sites)
    assert decision == 'formal_g1_retest' and stats['distinct_documents'] == 32
    assert stats['comparisons']['selected_vs_random']['all']['median_gain_pct'] < 0
    assert study.assess_documents(rows({11: 8., 23: 10.1}), SEEDS, sites)[1] == 'stop_batch_variant'
    uncertain, dec = study.assess_documents(rows({11: 9.9, 23: 9.9}, variable=True), SEEDS, sites)
    assert dec == 'stop_batch_variant' and uncertain['comparisons']['selected_vs_grid']['all']['paired_95_pct'][0] < 0
    assert study.assess_documents(rows({11: 8., 23: 9.}, grid=0.), SEEDS, sites)[1] == 'insufficient_evidence'


def test_integrity_verification_precedes_gpu_capture(tmp_path):
    source = tmp_path/'old'
    fit_receipt(source, 'parent')
    frozen = study.freeze_batch_inputs(tmp_path, source, ARCHIVE, SEEDS, 'parent')
    arms = load_matching_arms(ARCHIVE, SEEDS)
    old = atomic_json(tmp_path/'prior_documents.json', {'observed_01d': [
        {'document_id': d, 'document_sha256': 'hash'+d, 'token_ids_sha256': 'tokens'+d}
        for d in arms[11][CONTINUED]['heldout_documents']]})
    frozen['prior_documents_sha256'] = sha256_file(old)
    cfg = {'model_revision':'a'*40, 'tokenizer_revision':'b'*40, 'dataset_revision':'c'*40,
           'batch_document_pilot': frozen, 'seeds': SEEDS, 'formal_pilot': {'model_id':'fixture'},
           'rotation': {}, 'quantization': {}, 'solver': {}, 'runtime': {}}
    files = {}
    for s in SEEDS:
        rows = [{'document_id': f'{s}-{i}', 'document_sha256': f'd{s}-{i}', 'token_ids_sha256': f't{s}-{i}'} for i in range(16)]
        identity = {k:cfg[k] for k in ('model_revision','tokenizer_revision','dataset_revision')}
        files[str(s)] = str(atomic_json(tmp_path/f'seed_{s}.json', {'identity': {**identity, 'source_split':'train','seed':s},
                                                                    'partitions': {'dev': rows}}))
    protocol = atomic_json(tmp_path/'protocol.json', {**cfg, 'status':'FROZEN','model_id':'fixture',
                          'token_manifest_sha256':{s:sha256_file(p) for s,p in files.items()}})
    cfg.update(protocol_sha256=sha256_file(protocol), protocol_manifest_path=str(protocol), data={'token_manifests':files})
    study.verify_document_protocol(cfg)
    changed = copy.deepcopy(cfg)
    changed['runtime']['dtype'] = 'changed'
    with pytest.raises(ValueError, match='changed'):
        study.verify_document_protocol(changed)
    trace = tmp_path/'fit_reference/seed_11/batch_fit.json'
    trace.write_bytes(trace.read_bytes()+b' ')
    with pytest.raises(ValueError, match='checksum'):
        study.verify_document_protocol(cfg)
