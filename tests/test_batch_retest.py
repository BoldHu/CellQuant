from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip('torch')
from cellquant import batch_retest as retest
from cellquant.gates import load_rules


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'evidence/output_matching_20261119/batch_documents_7b.zip'


def test_promotion_is_reconstructed_from_real_blocks_and_token_hashes(tmp_path):
    result,stats,excluded = retest.read_document_pilot(BUNDLE)
    assert result['next_step'] == 'formal_g1_retest'
    assert stats['distinct_documents'] == 32 and len({b['document_id'] for b in excluded}) >= 32
    assert stats['comparisons']['selected_vs_grid']['all']['median_gain_pct'] == pytest.approx(.081199874683)
    assert stats['comparisons']['selected_vs_random']['all']['paired_95_pct'][0] < 0
    corrupt = tmp_path/'changed.zip'
    with ZipFile(BUNDLE) as original, ZipFile(corrupt,'w') as copy:
        for name in original.namelist():
            extra = b' ' if name=='run/seed_23/accepted_batches/document_result.json' else b''
            copy.writestr(name,original.read(name)+extra)
    with pytest.raises(ValueError,match='checksum'):
        retest.read_document_pilot(corrupt)


def test_retest_draws_both_fit_and_evaluation_after_excluding_all_earlier_cohorts():
    def block(i):
        return {'document_id':str(i),'document_sha256':f'd{i}','token_ids_sha256':f't{i}','block_id':f'{i}:0'}
    pool = [block(i) for i in range(160)]
    previous = {s:{'fit':pool[i:i+8],'dev':pool[i+8:i+16]} for s,i in ((11,0),(23,16))}
    old = pool[32:96]
    full = pool + [{**block(200),'document_sha256':'d72'},{**block(201),'token_ids_sha256':'t65'}]
    draws = retest.retest_partitions(full,previous,old)
    assert draws == retest.retest_partitions(full,previous,old)
    assert all(len(fit)==8 and len(dev)==16 for fit,dev in draws.values())
    for key in ('document_id','document_sha256','token_ids_sha256'):
        chosen = {b[key] for fit,dev in draws.values() for b in fit+dev}
        assert len(chosen)==48 and chosen.isdisjoint(b[key] for b in pool[:96])


def test_parent_bounds_use_the_frozen_two_routes_and_never_stop_at_equality():
    def bounds(error, cost):
        rows = [{'seed':s,'method':m,'total_calibration_seconds':t, 'heldout':[{'panel_id':'q','nmse':e}]}
                for s in (11,23) for m,t,e in ((retest.CONTINUED,100.,100.),(retest.CANDIDATE,cost,error))]
        return retest.route_bounds(rows,load_rules())
    assert bounds(99.9,102.)['both_routes_ruled_out']
    assert not bounds(90.,102.)['both_routes_ruled_out']
    assert not bounds(99.9,50.)['both_routes_ruled_out']


def test_gate_and_bootstrap_use_the_same_new_candidate_and_strong_parent():
    from cellquant.pilot_bootstrap import paired_document_intervals
    from cellquant.matching_study import HAD
    # Unit-test data, used only for the numeric interval routine.
    values = {HAD:2.,retest.CONTINUED:1.,retest.CANDIDATE:.99,
              **{m:1.5 for m in retest.CONTROLS}}
    records = [{'seed':s,'method':m,'panels':{'site':[{'document_id':str(s),'block_id':f'{s}:0',
                'sampled_tokens':32,'squared_error':v,'target_energy':1.}]}} for s in (11,23) for m,v in values.items()]
    d = {'seeds':[11,23],'sites':['site'],'new_nmse':[[.99],[.99]],
         'hadamard_nmse':[[2.],[2.]],'best_control_nmse':[[1.],[1.]]}
    stats = paired_document_intervals(records,d,repetitions=25,batch_retest=True)
    assert stats['comparisons']['best_matched_per_site']['median_site_gain_pct'] == pytest.approx(1.)
    assert retest.CONTINUED in stats['controls']


@pytest.mark.parametrize('complete,returncode',[(False,0),(True,2)])
def test_colab_retest_archives_a_bound_stop_or_a_resource_interruption(tmp_path,monkeypatch,complete,returncode):
    import importlib.util
    from cellquant.runtime import atomic_json
    spec = importlib.util.spec_from_file_location('retest_cli',ROOT/'scripts/run_output_matching.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setenv('PYTHONPATH','')
    parent = atomic_json(tmp_path/'prepared.json',{})
    def execute(argv,log,cwd,timeout_s):
        if '--output' in argv:
            assert '--retest-pilot' in argv
            atomic_json(Path(argv[argv.index('--output')+1])/'protocol.json',{})
            return {'returncode':0,'log_path':str(log)}
        assert '--batch-retest' in argv and ('--all-controls' in argv)==complete
        status = 'RESOURCE_BLOCKED' if returncode else 'RETEST_STOPPED_BY_BOUND'
        run=tmp_path/'out'
        atomic_json(run/'result.json',{'status':status,'complete_comparison':False,'route_bounds':{}})
        atomic_json(Path(argv[argv.index('--result-index')+1]),{'run_dir':str(run)})
        return {'returncode':returncode,'log_path':str(log)}
    monkeypatch.setattr(cli,'stream_command',execute)
    args=['--config',str(parent),'--retest-batches'] + (['--all-controls'] if complete else [])
    assert cli.main(args)==returncode
    package,=(tmp_path/'runs/01_pilot').rglob('batch_g1_retest.zip')
    with ZipFile(package) as z: assert 'run/result.json' in z.namelist() and 'inputs/protocol.json' in z.namelist()


def test_bilateral_fit_entry_recreates_stopped_cohort_without_running_the_old_solver(tmp_path, monkeypatch):
    import importlib.util
    from cellquant.runtime import atomic_json
    spec = importlib.util.spec_from_file_location('screen_fit_cli', ROOT/'scripts/run_output_matching.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    parent = atomic_json(tmp_path/'prepared.json', {})
    def study(argv, log, cwd, timeout_s):
        if '--output' in argv:
            assert argv[argv.index('--config')+1] == str(parent) and '--retest-pilot' in argv
            atomic_json(Path(argv[argv.index('--output')+1])/'config.json', {})
            return {'returncode':0, 'log_path':str(log)}
        assert '--screen-bilateral-angles' in argv and '--batch-retest' not in argv
        assert Path(argv[argv.index('--config')+1]).parent.name == 'inputs'
        out = tmp_path/'measured'
        atomic_json(out/'result.json', {'status':'DIAGNOSTIC_COMPLETE','source_type':'real_layer','next_step':'fresh_document_pilot'})
        atomic_json(Path(argv[argv.index('--result-index')+1]), {'run_dir':str(out)})
        return {'returncode':0, 'log_path':str(log)}
    monkeypatch.setattr(cli, 'stream_command', study)
    assert cli.main(['--screen-bilateral', '--config', str(parent)]) == 0
    package, = (tmp_path/'runs/01_pilot').rglob('bilateral_angle_screen.zip')
    with ZipFile(package) as archive:
        assert set(archive.namelist()) == {'inputs/config.json', 'run/result.json'}


def test_bilateral_screen_binds_the_stopped_cohorts_token_hashes(tmp_path, monkeypatch):
    import json
    from cellquant import experiment
    from cellquant.bilateral_screen import retest_grid_rows
    path = ROOT/'evidence/output_matching_20261119/batch_g1_retest_7b.zip'
    with ZipFile(path) as archive:
        cfg = json.loads(archive.read('run/config.json'))
        for name in ('protocol.json', 'prior_documents.json'):
            (tmp_path/name).write_bytes(archive.read('inputs/'+name))
        cfg['protocol_manifest_path'] = str(tmp_path/'protocol.json')
        for seed in cfg['seeds']:
            cached = tmp_path/f'{seed}.json'
            cached.write_bytes(archive.read(f'inputs/seed_{seed}/data_manifest.json'))
            cfg['data']['token_manifests'][str(seed)] = str(cached)
    rows = retest_grid_rows(path, cfg)
    assert all(row['solver']['completed_pairs'] == 256 for row in rows.values())
    monkeypatch.setattr(experiment, 'preflight', lambda *args: {'status':'READY', 'missing':[]})
    ready = experiment.run_experiment(cfg, project_root=tmp_path, preflight_only=True, bilateral_archive=path)
    assert ready['execution_mode'] == 'bilateral_angle_fit_screen' and ready['readiness']['status'] == 'READY'
    cached.write_bytes(b'{}')
    with pytest.raises(ValueError, match='articles'):
        retest_grid_rows(path, cfg)
