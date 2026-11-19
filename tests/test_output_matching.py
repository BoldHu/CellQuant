import itertools
import math
import importlib.util
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from cellquant.data import confirmation_partitions
from cellquant.matching_study import (DESIGN, PRODUCT, WEIGHT, WARM, attribution,
                                      paired_site_gains, run_matching_study)
from cellquant.output_matching import apply_matching, disjoint_pairs, fit_matching, fit_totals, pair_costs, proposal_grid, risk_tables
from cellquant.pilot import build_g1_evidence, panel_metrics
from cellquant.quantization import QuantSpec
from cellquant.runtime import atomic_json, read_json
from cellquant.solver import CalibrationPanel, SolverConfig, panel_specs
from cellquant.tensor_solver import PanelState, hadamard_rotation


def sample(seed, phase):
    rng = np.random.default_rng(seed)
    docs=[f'{seed}-{phase}-{i}' for i in range(3)]
    def panel(name, placement, width):
        w=rng.normal(size=(3,8) if placement=='input' else (8,width))
        x=rng.normal(size=(w.shape[1],6))
        return CalibrationPanel(name,placement,w,x,metadata={'source_type':'unit_test_fixture','layer':0,
            'document_ids':docs,'block_ids':[d+':0' for d in docs],'tokens_per_block':[2,2,2]})
    return [panel('up','input',8),panel('down_a','output',8),panel('down_b','output',12)]


@pytest.mark.parametrize('criterion',['weight','product'])
def test_cartesian_cost_equals_independent_pair_costs(criterion):
    fit=sample(13,'fit')
    q=hadamard_rotation(8,11,'cpu')
    specs=QuantSpec(group_size=4)
    pairs=disjoint_pairs(8,11)[:2]
    grid=proposal_grid(q,.68,3)
    table,coverage,out_delta=pair_costs(fit,q,specs,pairs,grid,criterion,batch_pairs=1)
    bounds=risk_tables(fit,q,specs,pairs,grid,out_delta,math.inf)
    base_risks=fit_totals(fit,q,specs)
    assert all(0 < bound < 1 for bound in coverage.values())
    def direct(new_q):
        vals=[]
        for p in fit[1:]:
            s=PanelState(p,new_q,*panel_specs(specs,p))
            vals.append(float(s.loss if criterion=='product' else (s.ah-s.a).square().sum()/s.wnorm))
        return sum(vals)
    base=direct(q)
    deltas=[]
    for j,k in itertools.product(range(len(grid)), repeat=2):
        rotated=apply_matching(q,pairs,grid[[j,k]])
        change=direct(rotated)-base
        expected=float(table[0,j]-table[0,0]+table[1,k]-table[1,0])
        assert change==pytest.approx(expected,abs=1e-12)
        actual=fit_totals(fit,rotated,specs)
        for key,bound in bounds.items():
            assert actual[key]-base_risks[key] <= float(bound[0,j]+bound[1,k])+1e-12
        deltas.append(change)
    assert min(deltas)==pytest.approx(float((table.min(dim=1).values-table[:,0]).sum()),abs=1e-12)
    with pytest.raises(ValueError,match='distinct'):
        apply_matching(q,[(1,2),(2,3)],grid[:2])


def test_rotation_joint_fit_risks_and_activation_invariant():
    fit=sample(22,'train')
    q=hadamard_rotation(8,11,'cpu')
    specs=QuantSpec(group_size=4)
    start=fit_totals(fit,q,specs)
    limits={k:start[k] for k in ('budget_w','budget_a')}
    new, info=fit_matching(fit,q,specs,disjoint_pairs(8,11),SolverConfig(),limits,criterion='product',grid_points=5)
    assert torch.linalg.norm(new.T@new-torch.eye(8)) < 1e-12
    assert info['final']==pytest.approx(fit_totals(fit,new,specs))
    assert info['final']['loss'] <= start['loss']
    assert all(info['final'][k] <= v+1e-10 for k,v in limits.items())
    full=info['joint_fit_trials'][-1]
    assert full['feasible']
    for k,v in info['risk_allocation']['full_step_risk_upper'].items():
        assert full[k] <= v+1e-12 and v <= limits[k]+1e-12
    # An arbitrary nonzero rotation makes the activation-only invariance test nontrivial.
    changed=apply_matching(q,disjoint_pairs(8,1),q.new_tensor([.2,.5,-.3,.4]))
    a,b=(attribution(panel_metrics(fit,m,specs,error_terms=True)) for m in (q,changed))
    for p in fit[1:]:
        assert b[p.panel_id]['activation_error_energy']==pytest.approx(a[p.panel_id]['activation_error_energy'],abs=1e-13)
    assert max(r['closure_abs_error'] for r in b.values())<1e-12


def test_study_pipeline_writes_raw_paired_evidence_without_a_gate(tmp_path):
    design={**DESIGN,'warm_pairs':2,'extra_grid_pairs':2,'sparse_pairs':1,
            'baseline_grid_points':5,'matching_grid_points':5,'bootstrap_draws':80}
    cfg={'seeds':[11,23],'solver':{'device':'cpu','backend':'torch','wall_seconds_per_method':100},
         'output_matching_study':{'design':design}}
    prepared={'hidden_size':8,'seeds':{s:{'fit':sample(s,'fit'),'heldout':sample(s+50,'dev'),
              'specs':QuantSpec(group_size=4),'capture_seconds':.1} for s in cfg['seeds']}}
    result=run_matching_study(cfg,prepared,tmp_path,execution_fingerprint='fixture')
    assert result['source_type']=='unit_test_fixture' and result['status']=='STUDY_COMPLETE'
    assert len(result['matching_artifacts'])==16 and not list(tmp_path.rglob('G1*.json'))
    rows=[read_json(p['path']) for p in result['matching_artifacts']]
    for seed in cfg['seeds']:
        same={r['method']:r for r in rows if r['seed']==seed}
        assert same[PRODUCT]['solver']['pairs']==same[WEIGHT]['solver']['pairs']
        assert same[PRODUCT]['solver']['limits']==same[WARM]['solver']['limits']
        assert len(same[PRODUCT]['solver']['pairs'])==4
    stats=result['matching_summary']
    assert stats['distinct_documents']==6 and 'output' in stats['comparisons']['primary_vs_warm_grid']
    assert 'hybrid_vs_continued_grid' in stats['comparisons']
    with pytest.raises(ValueError,match='Diagnostic results cannot form G1'):
        build_g1_evidence([result['matching_artifacts'][0]['path']],tmp_path/'not_g1',g0_evidence_path='unused')


def test_batch_screen_replays_fit_only_and_materializes_accepted_batches():
    from cellquant.matching_batches import screen_batches
    fit = sample(22, 'fit')
    q = hadamard_rotation(8, 11, 'cpu')
    specs = QuantSpec(group_size=4)
    cfg = SolverConfig(device='cpu')
    limits = {k: v for k, v in fit_totals(fit, q, specs).items() if k.startswith('budget_')}
    planes = disjoint_pairs(8, 11)
    _, ref = fit_matching(fit, q, specs, planes, cfg, limits, criterion='product', grid_points=5)
    result = screen_batches(fit, specs, q, ref, cfg, 11, batch_size=1)
    angles = np.asarray(ref['proposed_angles'])
    mask = np.zeros_like(angles)
    mask[result['accepted_batches']] = angles[result['accepted_batches']]
    expected = fit_totals(fit, apply_matching(q, planes, mask), specs)
    assert result['final'] == pytest.approx(expected)
    assert len(result['same_count_random']['batches']) == len(result['accepted_batches'])
    last = result['initial']['loss']
    for step in result['trace']:
        assert step['parent_loss'] == last
        assert step['accepted'] == (step['feasible'] and step['loss'] < last)
        if step['accepted']:
            last = step['loss']
    assert last == pytest.approx(expected['loss']) and len(result['trace']) == 4
    ref['joint_fit_trials'][-1]['output_loss'] += .01
    with pytest.raises(ValueError, match='replay differs'):
        screen_batches(fit, specs, q, ref, cfg, 11)


def test_batch_archive_checks_bytes_and_seed_and_discards_heldout(tmp_path):
    from cellquant.matching_batches import load_fit_reference
    source = Path(__file__).resolve().parents[1]/'evidence/output_matching_20261119/measurements.zip'
    records = load_fit_reference(source, [11, 23])
    assert all('heldout' not in arm for arms in records.values() for arm in arms.values())
    with pytest.raises(ValueError, match='missing'):
        load_fit_reference(source, [11, 23, 77])
    for corrupt in (False, True):
        archive = tmp_path/'colab.zip'
        with ZipFile(source) as original, ZipFile(archive, 'w') as wrapped:
            for name in original.namelist():
                suffix = b' ' if corrupt and name == 'seed_11/warm_grid/matching_result.json' else b''
                wrapped.writestr('run/'+name, original.read(name)+suffix)
        if corrupt:
            with pytest.raises(ValueError, match='checksum'):
                load_fit_reference(archive, [11, 23])
        else:
            assert load_fit_reference(archive, [11, 23]) == records


def test_batch_screen_uses_a_separate_fit_run_and_the_original_objective(tmp_path, monkeypatch):
    from cellquant import experiment
    source = Path(__file__).resolve().parents[1]/'evidence/output_matching_20261119/measurements.zip'
    cfg = {'stage': 'G1', 'runtime': {'model_device': 'cuda:0'}, 'result_path': str(tmp_path/'prior.json'),
           'solver': {'device': 'cuda:0', 'backend': 'torch', 'compute_dtype': 'float64'}}
    def ready(checks, root):
        assert 'result_path' not in checks
        return {'status': 'READY', 'missing': []}
    monkeypatch.setattr(experiment, 'preflight', ready)
    def preflight(archive):
        return experiment.run_experiment(cfg, project_root=tmp_path, preflight_only=True, matching_batch_archive=archive)
    current = preflight(source)
    assert current['readiness']['status'] == 'READY' and current['execution_mode'] == 'fit_matching_batch_screen'
    assert preflight(tmp_path/'missing.zip')['readiness']['status'] == 'BLOCKED'
    cfg['solver']['fit_objective'] = 'relative_site_nmse'
    assert preflight(source)['readiness']['status'] == 'BLOCKED'
    assert not Path(cfg['result_path']).exists()


def test_fresh_matching_validation_excludes_reserved_documents_and_content():
    def block(i):
        return {'document_id':str(i),'document_sha256':'h'+str(i),'token_ids_sha256':'t'+str(i)}
    pool=[block(i) for i in range(100)]
    prior={11:{'fit':pool[:8],'dev':pool[8:16]},23:{'fit':pool[16:24],'dev':pool[24:32]}}
    reserved=[b for _,dev in confirmation_partitions(pool,prior,8).values() for b in dev]
    duplicate={**block(101),'document_sha256':reserved[0]['document_sha256']}
    fresh=confirmation_partitions([duplicate,*pool],prior,8,namespace=DESIGN['partition_namespace'],exclude=reserved)
    new=[b for _,dev in fresh.values() for b in dev]
    assert len({b['document_id'] for b in new})==16
    for key in ('document_id','document_sha256','token_ids_sha256'):
        assert not {b[key] for b in new}&{b[key] for b in [*reserved,*pool[:32]]}


def test_preparation_binds_cached_weights_and_reuses_only_fit_documents(tmp_path,monkeypatch):
    from cellquant import data, experiment
    from cellquant.runtime import sha256_file
    datasets=pytest.importorskip('datasets')
    hub=pytest.importorskip('huggingface_hub')
    transformers=pytest.importorskip('transformers')
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('matching_prepare_test',root/'scripts/prepare_experiment.py')
    prep=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)

    # The only remote substitutes are the dataset rows, tokenizer, and cache lookup.
    class Corpus(list):
        _fingerprint='fixture_revision'
    corpus=Corpus({'text':f'= Article {i} ='} for i in range(128))
    def tokenizer(text,**kwargs):
        i=int(text.split()[2])
        return {'input_ids':[i]*2048}
    def load_tokenizer(repo_id,**kwargs):
        assert kwargs['local_files_only'] and kwargs['revision']=='b'*40
        return tokenizer
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',load_tokenizer)
    monkeypatch.setattr(datasets,'load_dataset',lambda *args,**kwargs:corpus)
    monkeypatch.setattr(prep,'resolve_revision',lambda *args,**kw:pytest.fail('Re-resolved a pinned revision'))
    monkeypatch.setattr(hub,'try_to_load_from_cache',lambda repo,name,**kw:str(tmp_path/name))
    monkeypatch.setattr(prep,'load_token_manifest',lambda p:data.load_token_manifest(p,allow_fixture=True))
    monkeypatch.setattr(prep,'save_token_manifest',lambda *args:data.save_token_manifest(*args,source_type='unit_test_fixture'))
    cfg=read_json(root/'configs/colab_pilot.json')
    cfg.update(model_revision='a'*40,tokenizer_revision='b'*40,dataset_revision='c'*40)
    cfg['runtime']['dtype']='float32'
    for name in ('weights.safetensors','config.json','tokenizer.json','tokenizer_config.json'):
        (tmp_path/name).write_bytes(b'unit_test_fixture')
    weights=tmp_path/'weights.safetensors'
    source=atomic_json(tmp_path/'model_source.json',{'model_id':cfg['formal_pilot']['model_id'],
        'revision':cfg['model_revision'],'files':[{'path':weights.name,'sha256':sha256_file(weights),'size_bytes':weights.stat().st_size}]})
    cfg.update(model_source_manifest_path=str(source),model_source_manifest_sha256=sha256_file(source))
    blocks=data.tokenize_documents(data.wikitext_documents(corpus),tokenizer,2048)
    original={}
    files={}
    for seed in cfg['seeds']:
        fit,dev=data.split_train_blocks(blocks,8,8,seed)
        original[seed]={'fit':fit,'dev':dev}
        identity={k:cfg[k] for k in ('model_revision','tokenizer_revision','dataset_revision')}
        identity.update(source_split='train',dataset_id=cfg['data']['calibration_dataset'],seed=seed,dataset_fingerprint=corpus._fingerprint)
        files[str(seed)]=str(prep.save_token_manifest(tmp_path/f'parent_{seed}',original[seed],identity))
    cfg['data']['token_manifests']=files
    protocol=atomic_json(tmp_path/'parent_protocol.json',{'token_manifest_sha256':{s:sha256_file(p) for s,p in files.items()}})
    cfg.update(protocol_manifest_path=str(protocol),protocol_sha256=sha256_file(protocol))
    prepared=read_json(prep.prepare(atomic_json(tmp_path/'parent.json',cfg),tmp_path/'fresh',matching_study=True))
    assert prepared['runtime']['local_cache_only'] and prepared['output_matching_study']['design']==DESIGN
    reserved={b['document_id'] for pair in data.confirmation_partitions(blocks,original,8).values() for b in pair[1]}
    new_docs=[]
    for seed in cfg['seeds']:
        _,parts=data.load_token_manifest(prepared['data']['token_manifests'][str(seed)],allow_fixture=True)
        assert [b['token_ids_sha256'] for b in parts['fit']]==[b['token_ids_sha256'] for b in original[seed]['fit']]
        new_docs.extend(b['document_id'] for b in parts['dev'])
    assert len(set(new_docs))==16 and not reserved.intersection(new_docs)
    monkeypatch.setattr(experiment,'preflight',lambda *args:{'status':'READY','missing':[]})
    measured=experiment.run_experiment(prepared,project_root=tmp_path,preflight_only=True,matching_study=True)
    assert measured['readiness']['status']=='READY'
    prepared['solver']['theta_limit']/=2
    changed=experiment.run_experiment(prepared,project_root=tmp_path,preflight_only=True,matching_study=True)
    assert changed['readiness']['status']=='BLOCKED'

    from cellquant import matching_batches
    from cellquant.batch_validation import verify_document_protocol
    from cellquant.matching_study import CONTINUED
    from cellquant.runtime import config_hash
    # Map the archive's document lookup onto this test corpus, keeping the original fit receipts.
    old_panels={}
    for seed in cfg['seeds']:
        _,old=data.load_token_manifest(prepared['data']['token_manifests'][str(seed)],allow_fixture=True)
        old_panels[seed]={CONTINUED:{'fit_documents':[b['document_id'] for b in old['fit']],
                                     'heldout':[{'blocks':old['dev']}]}}
    monkeypatch.setattr(matching_batches,'load_matching_arms',lambda archive,seeds:old_panels)
    screen=read_json(root/'evidence/output_matching_20261119/fit_batches_42m.json')
    for seed,trace in screen.pop('by_seed').items():
        atomic_json(tmp_path/'screen'/f'seed_{seed}'/'batch_fit.json',trace)
    atomic_json(tmp_path/'screen/result.json',{**screen,'run_id':'test_source','config_sha256':config_hash(cfg)})
    batch=read_json(prep.prepare(tmp_path/'parent.json',tmp_path/'batch_docs',batch_screen=tmp_path/'screen',
                    matching_archive=root/'evidence/output_matching_20261118/hybrid_42m/measurements.zip'))
    verify_document_protocol(batch)
    ids=set()
    for seed in cfg['seeds']:
        _,parts=data.load_token_manifest(batch['data']['token_manifests'][str(seed)],allow_fixture=True)
        assert [b['token_ids_sha256'] for b in parts['fit']]==[b['token_ids_sha256'] for b in original[seed]['fit']]
        ids.update(b['document_id'] for b in parts['dev'])
    assert len(ids)==32 and ids.isdisjoint(new_docs) and ids.isdisjoint(reserved)
    assert batch['formal_pilot']['heldout_tokens_per_panel']==512
    assert experiment.run_experiment(batch,project_root=tmp_path,preflight_only=True,batch_documents=True)['readiness']['status']=='READY'
    assert experiment.run_experiment(batch,project_root=tmp_path,preflight_only=True)['readiness']['status']=='BLOCKED'


def test_colab_terminal_entry_runs_preparation_and_only_the_registered_study(tmp_path,monkeypatch,capsys):
    path=Path(__file__).resolve().parents[1]/'scripts/run_output_matching.py'
    spec=importlib.util.spec_from_file_location('matching_cli_test',path)
    cli=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setenv('PYTHONPATH','')
    parent=atomic_json(tmp_path/'prepared.json',{})
    calls=[]
    def fake(argv,log,cwd,timeout_s):
        assert cwd==tmp_path and '--matching-study' in argv
        calls.append(Path(log).stem)
        if '--output' in argv:
            atomic_json(Path(argv[argv.index('--output')+1])/'config.json',{})
        else:
            run=tmp_path/'measured'
            effect={'median_gain_pct':1.25,'paired_95_pct':[-.1,2.5],'seed_medians_pct':{'11':2.5,'23':0}}
            study={'distinct_documents':16,'comparisons':{label:{side:effect for side in ('output','input','all')}
                   for label in ('primary_vs_warm_grid','activation_information','coverage','hybrid_vs_continued_grid')},
                   'hybrid_mechanism_by_seed':{'23':{'applied_pairs':0,'calibration_seconds':{
                       'continued_grid':10.,'continued_grid_plus_matching':10.4}}}}
            atomic_json(run/'result.json',{'source_type':'real_layer','status':'STUDY_COMPLETE','matching_summary':study})
            atomic_json(Path(argv[argv.index('--result-index')+1]),{'run_dir':str(run)})
        return {'returncode':0,'log_path':str(log)}
    monkeypatch.setattr(cli,'stream_command',fake)
    assert cli.main(['--config',str(parent)])==0 and calls==['prepare','study']
    captured=capsys.readouterr().out
    assert 'hybrid_vs_continued_grid / all: +1.25%' in captured and 'applied 0 pairs' in captured
    archive,=tmp_path.rglob('output_matching.zip')
    with ZipFile(archive) as z:
        assert set(z.namelist())=={'inputs/config.json','run/result.json'}
