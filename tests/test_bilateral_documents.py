import copy
import importlib.util
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip('torch')
from cellquant import bilateral_documents as docs, experiment
from cellquant.matching_study import HAD, paired_site_gains
from cellquant.runtime import atomic_json, config_hash, read_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT/'evidence/output_matching_20261119'


def test_real_screen_traces_and_frozen_cohort_are_bound_to_both_seeds(tmp_path):
    screen, grid = REFS/docs.FIT_FILE, REFS/docs.GRID_FILE
    fit = docs.read_7b_fit(screen, grid)
    cohort = read_json(REFS/docs.COHORT_FILE)
    assert fit['receipt']['next_step'] == 'fresh_document_pilot' and sha256_file(screen) == cohort['fit_archive_sha256']
    old = [*fit['prior'], *cohort['previous_42m_train_blocks'], *(b for p in fit['parts'].values() for s in p.values() for b in s)]
    chosen = [b for rows in cohort['heldout'].values() for b in rows]
    for key in docs.IDENTITY:
        assert len({b[key] for b in chosen}) == 32
        assert not {b[key] for b in chosen}.intersection(b[key] for b in old)
    other = tmp_path/'changed.zip'
    with ZipFile(screen) as z, ZipFile(other,'w') as dest:
        for k in z.namelist():
            dest.writestr(k,z.read(k) + (b' ' if k.endswith('seed_23/angle_screen.json') else b''))
    with pytest.raises(ValueError,match='checksum'):
        docs.read_7b_fit(other,grid)
    # Use real archived block identities to exercise an altered parent-token input.
    blocks = [dict(b) for p in fit['parts'].values() for rows in p.values() for b in rows]
    blocks[0]['token_ids_sha256'] = 'changed'
    with pytest.raises(ValueError,match='tokenizer'):
        docs.draw_documents(blocks,fit,[])


@pytest.mark.parametrize('study',['documents','crossfit','refresh','transfer'])
def test_preflight_binds_fresh_documents_and_dispatches_only_the_document_mode(tmp_path, monkeypatch, study):
    refs = tmp_path/'fit_reference'
    refs.mkdir()
    names = (docs.FIT_FILE,docs.GRID_FILE,docs.COHORT_FILE)
    for name in names:
        shutil.copyfile(REFS/name,refs/name)
    cohort = read_json(refs/docs.COHORT_FILE)
    if study != 'documents':
        from cellquant.row_crossfit import COHORT_FILE, DESIGN
        shutil.copyfile(REFS/COHORT_FILE,refs/COHORT_FILE)
        cohort = read_json(refs/COHORT_FILE)
    if study in ('refresh','transfer'):
        from cellquant.row_refresh import COHORT_FILE as refresh_file, DESIGN as refresh_design, REFERENCE_FILE
        for name in (refresh_file,REFERENCE_FILE):
            shutil.copyfile(REFS/name,refs/name)
        cohort = read_json(refs/refresh_file)
    if study == 'transfer':
        from cellquant.row_transfer import REFERENCE_FILE as transfer_file, transfer_design
        shutil.copyfile(REFS/transfer_file,refs/transfer_file)
    with ZipFile(refs/docs.FIT_FILE) as z:
        cfg = json.loads(z.read('run/config.json'))
        parent = cfg.pop('batch_g1_retest')
        cfg['bilateral_document_pilot'] = {'design':docs.DESIGN, 'archives':{k:sha256_file(refs/k) for k in names},
            'quality_rules_sha256':config_hash(docs.load_rules()),'parent_config_sha256':parent['parent_config_sha256']}
        files = {}
        for s in cfg['seeds']:
            m = json.loads(z.read(f'inputs/seed_{s}/data_manifest.json'))
            m['partitions']['dev'] = cohort['heldout'][str(s)]
            files[str(s)] = str(atomic_json(tmp_path/f'seed_{s}.json',m))
        protocol = json.loads(z.read('inputs/protocol.json'))
        protocol.pop('batch_g1_retest')
        protocol.update(bilateral_document_pilot=cfg['bilateral_document_pilot'],token_manifest_sha256={k:sha256_file(v) for k,v in files.items()})
        if study != 'documents':
            cfg['row_crossfit'] = protocol['row_crossfit'] = {'design':DESIGN,'cohort_sha256':sha256_file(refs/COHORT_FILE)}
        if study in ('refresh','transfer'):
            cfg['row_refresh'] = protocol['row_refresh'] = {'design':refresh_design,'cohort_sha256':sha256_file(refs/refresh_file),
                                                            'crossfit_sha256':sha256_file(refs/REFERENCE_FILE)}
        if study == 'transfer':
            cfg['row_transfer'] = protocol['row_transfer'] = {'design':transfer_design(),'refresh_sha256':sha256_file(refs/transfer_file)}
    path = atomic_json(tmp_path/'protocol.json',protocol)
    cfg.update(protocol_manifest_path=str(path),protocol_sha256=sha256_file(path))
    cfg['data']['token_manifests'] = files
    monkeypatch.setattr(experiment,'preflight',lambda *a: {'status':'READY','missing':[]})
    assert experiment.run_experiment(cfg,project_root=tmp_path,preflight_only=True)['readiness']['status'] == 'BLOCKED'
    mode = {'documents':'frozen_bilateral_document_pilot','crossfit':'matched_row_crossfit','refresh':'matched_row_refresh',
            'transfer':'frozen_cross_layer_row_transfer'}[study]
    flags = dict(bilateral_documents=study=='documents',row_crossfit=study=='crossfit',row_refresh=study=='refresh',row_transfer=study=='transfer')
    out = experiment.run_experiment(cfg,project_root=tmp_path,preflight_only=True,**flags)
    assert out['execution_mode'] == mode and out['readiness']['status'] == 'READY'
    bad = read_json(files['23'])
    bad['partitions']['dev'][0] = bad['partitions']['fit'][0]
    atomic_json(files['23'],bad)
    protocol['token_manifest_sha256']['23'] = sha256_file(files['23'])
    cfg['protocol_sha256'] = sha256_file(atomic_json(path,protocol))
    with pytest.raises(ValueError,match='identity'):
        docs.verify_document_inputs(cfg,row_crossfit=flags['row_crossfit'],row_refresh=flags['row_refresh'],row_transfer=flags['row_transfer'])


def test_document_decision_separates_mechanism_from_grid_tail_and_efficiency():
    watch = docs.DESIGN['watch_site']
    s = {'x':'output',watch:'input','z':'input'}
    rows = [{'seed':seed,'method':m,'panels':{site:[
                {'document_id':d,'block_id':d+':0','sampled_tokens':32,'squared_error':err,'target_energy':1.}
                for d in (str(seed)+'a',str(seed)+'b')] for site in s}}
            for seed in [11,23] for m,err in [('candidate',.99),('grid',1.),('old',1.05),(HAD,1.2)]]
    stats = paired_site_gains(rows,[11,23],s,{'vs_output_only':('candidate','old'),'vs_full_grid':('candidate','grid'),
                            'vs_hadamard':('candidate',HAD)},repetitions=50,watched_sites=[watch])
    stats['mean_site_nmse_by_seed'] = {str(s):{'bilateral_fixed_scale_proposal':.99,HAD:1.2} for s in (11,23)}
    verdict = docs.document_decision(stats)
    assert verdict['next_step'] == 'formal_g1_retest' and not verdict['failed_quality_bounds_by_route']['efficiency']
    assert verdict['watched_site_vs_grid']['paired_95_pct'] == pytest.approx([1.,1.])
    inconclusive = copy.deepcopy(stats)
    inconclusive['comparisons']['vs_output_only']['all']['paired_95_pct'][0] = -1.
    assert docs.document_decision(inconclusive)['next_step'] == 'review_inconclusive_pilot'
    stats['comparisons']['vs_full_grid']['site_gain_pct'][watch] = -10.1
    verdict = docs.document_decision(stats)
    assert verdict['mechanism_replicated'] and verdict['next_step'] == 'stop_bilateral_variant'
    assert all('max_site_nmse_regression_vs_best_pct' in v for v in verdict['failed_quality_bounds_by_route'].values())
    stats['comparisons']['vs_full_grid'] = {'undefined':'zero comparator'}
    assert docs.document_decision(stats)['next_step'] == 'insufficient_evidence'


def test_terminal_accepts_the_portable_fit_zip_without_an_old_g1_config(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('docs_colab_cli', ROOT/'scripts/run_output_matching.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setenv('PYTHONPATH','')
    complete = False
    def execute(argv,log,cwd,timeout_s):
        if '--output' in argv:
            assert argv[argv.index('--config')+1] == str(REFS/docs.FIT_FILE)
            assert argv[argv.index('--bilateral-sources')+1] == str(REFS/docs.FIT_FILE)
            atomic_json(Path(argv[argv.index('--output')+1])/'config.json',{})
            return {'returncode':0,'log_path':str(log)}
        assert '--bilateral-documents' in argv and '--screen-bilateral-angles' not in argv
        run=tmp_path/'out'
        outcome = {'status':'RESOURCE_BLOCKED','source_type':'not_run','reason':'fixture'}
        if complete:
            outcome = {'status':'STUDY_COMPLETE','source_type':'unit_test_fixture'}
        atomic_json(run/'result.json',outcome)
        atomic_json(Path(argv[argv.index('--result-index')+1]),{'run_dir':str(run)})
        return {'returncode':0 if complete else 2,'log_path':str(log)}
    monkeypatch.setattr(cli,'stream_command',execute)
    assert cli.main(['--validate-bilateral','--fit-archive',str(REFS/docs.FIT_FILE)]) == 2
    artifact, = (tmp_path/'runs/01_pilot').rglob('bilateral_document_pilot.zip')
    with ZipFile(artifact) as z:
        assert json.loads(z.read('run/result.json'))['status'] == 'RESOURCE_BLOCKED'
    complete = True
    with pytest.raises(RuntimeError,match='Expected completed real bilateral'):
        cli.main(['--validate-bilateral','--fit-archive',str(REFS/docs.FIT_FILE)])


@pytest.mark.parametrize('study',['diagnosis','crossfit','refresh','transfer'])
@pytest.mark.parametrize('migrate',[False,True])
def test_row_command_reuses_portable_archives_and_preserves_an_incomplete_run(tmp_path, monkeypatch,study,migrate):
    spec = importlib.util.spec_from_file_location('row_colab_cli', ROOT/'scripts/run_output_matching.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setenv('PYTHONPATH','')
    refs = tmp_path/'evidence/output_matching_20261119'
    refs.mkdir(parents=True)
    document_zip = refs/'bilateral_documents_7b.zip'
    shutil.copyfile(REFS/document_zip.name, document_zip)

    def execute(argv, log, cwd, timeout_s):
        if '--output' in argv:
            if migrate:
                assert argv[argv.index('--cache-dir')+1]==str((tmp_path/'cache/hub').resolve())
                assert argv[argv.index('--dataset-cache-dir')+1]==str((tmp_path/'cache/datasets').resolve())
                assert '--allow-model-download' in argv
            else:
                assert '--cache-dir' not in argv and '--allow-model-download' not in argv
            expected = [str(refs/n) for n in (docs.FIT_FILE,docs.GRID_FILE,docs.COHORT_FILE)]
            if study != 'diagnosis':
                from cellquant.row_crossfit import COHORT_FILE
                expected.extend(['--row-crossfit-cohort',str(refs/COHORT_FILE)])
            if study in ('refresh','transfer'):
                from cellquant.row_refresh import COHORT_FILE, REFERENCE_FILE
                expected.extend(['--row-refresh-sources',str(refs/REFERENCE_FILE),str(refs/COHORT_FILE)])
            if study == 'transfer':
                from cellquant.row_transfer import REFERENCE_FILE as transfer_file
                expected.extend(['--row-transfer-archive',str(refs/transfer_file)])
            assert argv[argv.index('--bilateral-sources')+1:] == expected
            atomic_json(Path(argv[argv.index('--output')+1])/'config.json', {})
            return {'returncode':0, 'log_path':str(log)}
        if study == 'transfer':
            assert '--audit-row-transfer' in argv and '--refresh-rows' not in argv
        elif study == 'refresh':
            assert '--refresh-rows' in argv and '--crossfit-rows' not in argv
        elif study == 'crossfit':
            assert '--crossfit-rows' in argv and '--bilateral-rows' not in argv
        else:
            assert argv[argv.index('--bilateral-rows')+1] == str(document_zip)
        run = tmp_path/'out'
        atomic_json(run/'result.json', {'status':'RESOURCE_BLOCKED', 'source_type':'not_run', 'reason':'test fixture'})
        atomic_json(Path(argv[argv.index('--result-index')+1]), {'run_dir':str(run)})
        return {'returncode':2, 'log_path':str(log)}

    monkeypatch.setattr(cli,'stream_command',execute)
    flag,filename = {'refresh':('--refresh-rows','row_refresh.zip'),'crossfit':('--crossfit-rows','row_crossfit.zip'),
                     'diagnosis':('--diagnose-bilateral-rows','bilateral_row_diagnostic.zip'),
                     'transfer':('--audit-row-transfer','row_transfer.zip')}[study]
    overrides=['--cache-dir',str(tmp_path/'cache/hub'),'--dataset-cache-dir',str(tmp_path/'cache/datasets'),
               '--allow-model-download'] if migrate else []
    assert cli.main([flag,*overrides]) == 2
    path, = (tmp_path/'runs/01_pilot').rglob(filename)
    with ZipFile(path) as z:
        assert json.loads(z.read('run/result.json'))['status'] == 'RESOURCE_BLOCKED'
