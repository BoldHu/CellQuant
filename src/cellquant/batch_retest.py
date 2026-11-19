"""Confirm the fitted batch rule on independent calibration and evaluation articles."""
import hashlib
import json
import math
import time
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from . import batch_validation as study
from .data import confirmation_partitions, token_hash
from .matching_batches import BATCH_SIZE, q_digest, screen_batches
from .matching_study import CONTINUED, HAD
from .output_matching import apply_matching, disjoint_pairs, fit_matching
from .pilot import BATCH_CANDIDATE as CANDIDATE, build_g1_evidence, panel_metrics, pilot_solver_config, summarize_pilot_results
from .progress import report
from .runtime import atomic_json, config_hash, environment_report, read_json, sha256_file
from .solver import atomic_npz, pair_schedule, panel_fingerprint
from .tensor_solver import fit_baseline, hadamard_rotation, synchronize


CONTROLS = ('angle_grid_matched', 'givens_ste_matched', 'fixed_scale_search_matched')
DESIGN = {'version': 1, 'partition_namespace': 2026111903, 'seeds': [11, 23], 'fit': 8, 'dev': 16,
          'prefix_pairs': 128, 'extra_grid_pairs': 128, 'parent_grid_points': 129, 'matching_grid_points': 17,
          'batch_size': BATCH_SIZE, 'step_scales': [0., .25, .5, 1.], 'fit_objective': 'sum_nmse',
          'candidate': CANDIDATE, 'controls': [CONTINUED, *CONTROLS],
          'control_schedule': 'same continued grid, disjoint axes, angle range, risk caps and total time budget'}


def read_document_pilot(bundle):
    """Recompute the promotion from the submitted document measurements, including their identities."""
    with ZipFile(bundle) as archive:
        def read(name):
            return json.loads(archive.read(name))
        def checked(name, digest):
            payload = archive.read(name)
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f'Document pilot checksum mismatch: {name}')
            return payload
        result = read('run/result.json')
        config = read('run/config.json')
        source = result['fit_source']
        protocol = json.loads(checked('inputs/protocol.json', result['protocol_sha256']))
        fit_receipt = json.loads(checked('inputs/fit_reference/result.json', source['screen_result_sha256']))
        checked('inputs/fit_reference/measurements.zip', source['archive_sha256'])
        prior = json.loads(checked('inputs/prior_documents.json', source['prior_documents_sha256']))
        if (result['source_type'] != 'real_layer' or result['status'] != 'STUDY_COMPLETE'
                or result['evidence_kind'] != study.KIND or result['design'] != study.DESIGN
                or protocol['batch_document_pilot'] != source or source['design'] != study.DESIGN
                or config_hash(config) != result['config_sha256'] or config['seeds'] != DESIGN['seeds']
                or fit_receipt['config_sha256'] != source['parent_config_sha256']
                or fit_receipt['archive_sha256'] != source['archive_sha256']):
            raise ValueError('Unexpected document pilot protocol or source')
        keys = ('document_id', 'document_sha256', 'token_ids_sha256')
        excluded = [b for rows in prior.values() for b in rows]
        seen = {k: {b[k] for b in excluded} for k in keys}
        records, placements, docs, traces = [], {}, {}, {}
        for seed in config['seeds']:
            sub = f'inputs/seed_{seed}'
            manifest = json.loads(checked(sub+'/data_manifest.json', protocol['token_manifest_sha256'][str(seed)]))
            identity = manifest['identity']
            if (identity['seed'] != seed or identity['source_split'] != 'train' or identity['model_id'] != protocol['model_id']
                    or any(identity[k] != protocol[k] for k in ('model_revision','tokenizer_revision','dataset_revision'))):
                raise ValueError('Document measurement provenance mismatch')
            with np.load(BytesIO(checked(sub+'/token_ids.npz', manifest['tokens_sha256'])), allow_pickle=False) as tokens:
                for rows in manifest['partitions'].values():
                    for b in rows:
                        if token_hash(tokens[b['array_key']]) != b['token_ids_sha256']:
                            raise ValueError('Document pilot token cache mismatch')
            rows = manifest['partitions']['dev']
            if len(rows) != study.DESIGN['documents_per_seed']:
                raise ValueError('Unexpected number of evaluation articles')
            for k in keys:
                values = {b[k] for b in rows}
                if len(values) != len(rows) or values & seen[k]:
                    raise ValueError(f'Document pilot reused {k}')
                seen[k].update(values)
            excluded.extend(rows)
            docs[seed] = {b['block_id']: b['document_id'] for b in rows}
            trace = json.loads(checked(f'inputs/fit_reference/seed_{seed}/batch_fit.json',
                                       fit_receipt['fit_screen'][str(seed)]['fit_trace_sha256']))
            if trace['seed'] != seed or trace['archive_sha256'] != source['archive_sha256']:
                raise ValueError('Fit trace identity mismatch')
            traces[seed] = trace
        panel_hashes = {s: set() for s in config['seeds']}
        for ref in result['document_artifacts']:
            row = json.loads(checked('run/'+ref['path'], ref['sha256']))
            seed, method = row['seed'], row['method']
            if (ref['path'] != f'seed_{seed}/{method}/document_result.json' or row['source_type'] != 'real_layer'
                    or row['evidence_kind'] != study.KIND or method not in (study.SELECTED, CONTINUED, study.RANDOM)):
                raise ValueError('Unexpected pilot measurement')
            trace = traces[seed]
            expected_q = {study.SELECTED: trace['final_q_sha256'], CONTINUED: trace['parent_q_sha256']}
            if method in expected_q and row['q_sha256'] != expected_q[method]:
                raise ValueError('Frozen Q checksum mismatch')
            panel_hashes[seed].add(row['heldout_panel_sha256'])
            panels = {}
            for p in row['heldout']:
                blocks = p['blocks']
                if ({b['block_id']: b['document_id'] for b in blocks} != docs[seed]
                        or any(b['sampled_tokens'] != study.DESIGN['tokens_per_document'] for b in blocks)):
                    raise ValueError('Block measurements differ from the frozen token manifest')
                nmse = sum(b['squared_error'] for b in blocks) / sum(b['target_energy'] for b in blocks)
                if not math.isclose(nmse, p['nmse'], rel_tol=1e-10):
                    raise ValueError('Block sums disagree with the reported NMSE')
                panels[p['panel_id']] = blocks
                if placements.setdefault(p['panel_id'], p['placement']) != p['placement']:
                    raise ValueError('Placement mismatch')
            records.append({'seed': seed, 'method': method, 'panels': panels})
        if len(records) != 6 or len(placements) != 9 or any(len(h) != 1 for h in panel_hashes.values()):
            raise ValueError('Incomplete paired pilot')
        stats, decision = study.assess_documents(records, config['seeds'], placements)
        if decision != result['next_step'] or stats['distinct_documents'] != result['document_summary']['distinct_documents']:
            raise ValueError('Document pilot decision disagrees with the raw measurements')
        for label, comparison in stats['comparisons'].items():
            if 'undefined' in comparison:
                if comparison != result['document_summary']['comparisons'][label]:
                    raise ValueError(f'Document pilot undefined contrast differs for {label}')
                continue
            for side in ('all','input','output'):
                measured = comparison[side]
                original = result['document_summary']['comparisons'][label][side]
                for key in ('median_gain_pct', 'paired_95_pct'):
                    if not np.allclose(measured[key], original[key], rtol=1e-10, atol=1e-11):
                        raise ValueError(f'Document pilot statistic differs for {label}, {side}, {key}')
                if any(not math.isclose(v,original['seed_medians_pct'][s],rel_tol=1e-10,abs_tol=1e-11)
                       for s,v in measured['seed_medians_pct'].items()):
                    raise ValueError(f'Document pilot seed medians differ for {label}, {side}')
        return result, stats, [{k:b[k] for k in (*keys,'block_id')} for b in excluded]


def retest_partitions(blocks, previous, exclusions):
    count = DESIGN['fit'] + DESIGN['dev']
    drawn = confirmation_partitions(blocks, previous, count, namespace=DESIGN['partition_namespace'], exclude=exclusions)
    return {s: (articles[:DESIGN['fit']], articles[DESIGN['fit']:]) for s, (_, articles) in drawn.items()}


def verify_protocol(config):
    from .gates import resolve_rules_path
    from .objective_screen import verify_parent_manifests
    verify_parent_manifests(config)
    path = Path(config['protocol_manifest_path'])
    protocol = read_json(path)
    spec = config['batch_g1_retest']
    keys = ('batch_g1_retest','formal_pilot','rotation','quantization','solver','runtime',
            'dataset_revision','model_revision','tokenizer_revision')
    prior = path.parent/'prior_documents.json'
    if (spec['design'] != DESIGN or config['seeds'] != DESIGN['seeds'] or protocol['status'] != 'FROZEN'
            or protocol['model_id'] != config['formal_pilot']['model_id']
            or spec['gate_rules_sha256'] != sha256_file(resolve_rules_path())
            or sha256_file(prior) != spec['excluded_sha256'] or any(protocol[k] != config[k] for k in keys)):
        raise ValueError('The batch G1 retest differs from its frozen protocol')
    fields = ('document_id','document_sha256','token_ids_sha256')
    seen = {k: {r[k] for r in read_json(prior)} for k in fields}
    for seed in config['seeds']:
        manifest = read_json(config['data']['token_manifests'][str(seed)])
        identity = manifest['identity']
        if (identity['source_split'] != 'train' or identity['seed'] != seed or
                any(identity[k] != config[k] for k in ('model_revision','tokenizer_revision','dataset_revision'))):
            raise ValueError('Retest token provenance differs')
        for split in ('fit','dev'):
            rows = manifest['partitions'][split]
            for k in fields:
                chosen = {r[k] for r in rows}
                if len(rows) != DESIGN[split] or len(chosen) != len(rows) or chosen & seen[k]:
                    raise ValueError(f'Retest {split} documents reused {k}')
                seen[k].update(chosen)


def route_bounds(records, rules):
    """The best per-site comparator has at most parent NMSE; the fastest takes at most parent time."""
    seeds = sorted({r['seed'] for r in records})
    by = {(r['seed'],r['method']):r for r in records}
    gains, ratios = [], []
    for s in seeds:
        parent, candidate = (by[s,m] for m in (CONTINUED,CANDIDATE))
        p = {v['panel_id']:v['nmse'] for v in parent['heldout']}
        c = {v['panel_id']:v['nmse'] for v in candidate['heldout']}
        if p.keys() != c.keys() or not p or any(v <= 0 for v in p.values()):
            raise ValueError('Parent bound requires paired sites with positive comparator NMSE')
        gains.append([100*(1-c[k]/p[k]) for k in sorted(p)])
        ratios.append(parent['total_calibration_seconds']/candidate['total_calibration_seconds'])
    upper_gain = float(np.median(np.mean(gains,axis=0)))
    upper_speed = float(np.mean(ratios))
    gate = next(g for g in rules['gates'] if g['gate_id'] == 'G1')
    limits = {c['metric']:c['value'] for route in gate['routes'].values() for c in route}
    accuracy = limits['median_nmse_gain_vs_best_matched_solver_pct']
    speed = limits['pilot_calibration_speedup']
    return {'accuracy_upper_gain_pct':upper_gain,'accuracy_minimum_pct':accuracy,
            'efficiency_upper_speedup':upper_speed,'efficiency_minimum_speedup':speed,
            'both_routes_ruled_out':upper_gain < accuracy and upper_speed < speed}


def remaining(cfg, seconds):
    from .events import ResourceLimit
    if seconds >= cfg.wall_seconds:
        raise ResourceLimit('The common calibration wall budget is exhausted')
    return replace(cfg, wall_seconds=cfg.wall_seconds-seconds)


def fit_parent(panels, q0, specs, schedule, cfg, design, capture_cost, progress):
    from .events import ResourceLimit
    warm = design['prefix_pairs']
    q, limits, accumulated = q0, None, capture_cost
    steps = []
    count = design['parent_grid_points']
    for axes in (schedule[:warm], schedule[warm:]):
        solved = fit_baseline('angle_grid_matched', panels, q, specs, axes, remaining(cfg, accumulated),
                              grid_points=count, budgets=limits, progress=progress)
        q = solved.pop('q')
        accumulated += solved['seconds']
        if (solved['status'] != 'COMPLETE' or solved['completed_pairs'] != len(axes)
                or any(t['evaluations'] != count for t in solved['log']) or accumulated > cfg.wall_seconds):
            raise ResourceLimit('The common parent did not complete both full angle grids inside budget')
        limits = solved['limits']
        steps.append(solved)
    return q, limits, accumulated, steps


def fit_candidate(panels, specs, parent, pairs, limits, cfg, parent_cost, seed, design, progress):
    _, proposal = fit_matching(panels, parent, specs, pairs, remaining(cfg,parent_cost), limits, criterion='product',
                               grid_points=design['matching_grid_points'], step_scales=design['step_scales'], progress=progress)
    spent = parent_cost + proposal['seconds']
    trace = screen_batches(panels, specs, parent, proposal, remaining(cfg,spent), seed,
                           batch_size=design['batch_size'], progress=progress)
    angles = parent.new_tensor(proposal['proposed_angles'])
    chosen = parent
    mask = angles.new_zeros(len(angles))
    for index in trace['accepted_batches']:
        start = index*design['batch_size']
        block = slice(start,start+design['batch_size'])
        chosen = apply_matching(chosen, pairs[block], angles[block])
    for index in trace['same_count_random']['batches']:
        start = index*design['batch_size']
        mask[start:start+design['batch_size']] = angles[start:start+design['batch_size']]
    if q_digest(chosen) != trace['final_q_sha256']:
        raise ValueError('Replayed candidate differs from the fitted global rotation')
    return chosen, apply_matching(parent,pairs,mask), spent+trace['fit_screen_seconds'], proposal, trace


def run_retest(config, prepared, output, graph_identity, *, all_controls=False, progress=None, engineering_pairs=None):
    from .events import ResourceLimit
    from .gates import load_rules
    from .pilot_bootstrap import analyze_run
    output = Path(output)
    design = config['batch_g1_retest']['design']
    if design != DESIGN and engineering_pairs is None:
        raise ValueError('Formal G1 requires the registered schedule')
    cfg = pilot_solver_config(config)
    device = cfg.device
    env = environment_report()
    hardware = config_hash({k:env[k] for k in ('python','packages','cuda','platform')})
    paths, records, contexts = [], [], {}

    def save(seed, method, q, total, solver, common, ready, *, explanatory=False):
        if solver.get('status') == 'COMPLETE' and total > cfg.wall_seconds:
            solver = {**solver, 'status':'RESOURCE_BLOCKED', 'reason':'Total calibration wall budget exhausted'}
        dest = output/f'seed_{seed}'/method
        atomic_npz(dest/'Q.npz', Q=q)
        report(progress, 'retest.score', seed=seed, method=method)
        metrics = panel_metrics(ready['heldout'], q, ready['specs'], device, progress=progress,
                                error_terms=explanatory) if solver['status']=='COMPLETE' else []
        row = {**common,'method':method,'status':solver['status'],'heldout':metrics,'total_calibration_seconds':total,
               'q_path':str((dest/'Q.npz').resolve()), 'q_sha256':sha256_file(dest/'Q.npz'),
               'solver':{**solver,'seconds':total,'shared_global_r1':True,'fit_objective':{'mode':'sum_nmse'}}}
        path = atomic_json(dest/('random_result.json' if explanatory else 'pilot_result.json'),row)
        records.append(row)
        if not explanatory:
            paths.append(path)

    for seed in config['seeds']:
        report(progress,'retest.fit_parent',seed=seed)
        ready = prepared['seeds'][seed]
        fit, specs = ready['fit'],ready['specs']
        if any(p.metadata.get('source_type')!='real_layer' for p in [*fit,*ready['heldout']]):
            raise ValueError('Formal batch retest requires real layer captures')
        manifest = Path(config['data']['token_manifests'][str(seed)])
        parts = read_json(manifest)['partitions']
        fit_docs = sorted(b['document_id'] for b in parts['fit'])
        dev_docs = sorted(b['document_id'] for b in parts['dev'])
        for panels, ids in ((fit, fit_docs),(ready['heldout'],dev_docs)):
            if any(sorted(p.metadata['document_ids']) != ids for p in panels):
                raise ValueError('Captured documents differ from the prepared retest')
        n = prepared['hidden_size']
        schedule = pair_schedule(n,design['prefix_pairs']+design['extra_grid_pairs'],seed)
        pairs = disjoint_pairs(n,seed)
        if engineering_pairs is not None:
            if not 1 <= engineering_pairs <= len(pairs):raise ValueError('Invalid engineering pair limit')
            pairs = pairs[:engineering_pairs]
        synchronize(device)
        start = time.perf_counter()
        q0 = hadamard_rotation(n,seed,device)
        synchronize(device)
        initial_cost = ready['capture_seconds']+(time.perf_counter()-start)
        parent, limits, cost, steps = fit_parent(fit,q0,specs,schedule,cfg,design,initial_cost,progress)
        common = {'source_type':'real_layer','pilot_variant':CANDIDATE,'seed':seed,
            'evidence_kind':'BATCH_RETEST_ENGINEERING' if engineering_pairs is not None else 'FORMAL_BATCH_G1',
            'hidden_size':n,
            'model_id':config['formal_pilot']['model_id'],'model_revision':config['model_revision'],
            'model_parameter_billions':prepared['model_parameter_billions'],'quantization':config['quantization'],
            'graph_identity':graph_identity,'hardware_sha256':hardware,'wall_budget_seconds':cfg.wall_seconds,
            'schedule_sha256':config_hash({'prefix':schedule,'matching':pairs.tolist()}),
            'fit_panel_fingerprint':panel_fingerprint(fit),'heldout_panel_fingerprint':panel_fingerprint(ready['heldout']),
            'data_manifest_path':str(manifest.resolve()),'data_manifest_sha256':sha256_file(manifest),
            'fit_documents':fit_docs,'heldout_documents':dev_docs,'capture_seconds':ready['capture_seconds'],
            'context_lengths':[read_json(manifest)['identity']['context_length']],'parent_q_sha256':q_digest(parent)}
        save(seed,HAD,q0,initial_cost,{'status':'COMPLETE','factors':[]},common,ready)
        save(seed,CONTINUED,parent,cost,{'status':'COMPLETE','requested_pairs':len(schedule),
             'completed_pairs':len(schedule),'grid_stages':steps},common,ready)
        chosen, random, total, proposal, trace = fit_candidate(fit,specs,parent,pairs,limits,cfg,cost,seed,design,progress)
        if total > cfg.wall_seconds:
            raise ResourceLimit('Batch fitting exceeded the total calibration budget')
        save(seed,CANDIDATE,chosen,total,{'status':'COMPLETE','requested_pairs':len(pairs),
             'completed_pairs':len(pairs),'proposal':proposal,'selection':trace},common,ready)
        save(seed,study.RANDOM,random,total,{'status':'COMPLETE','count_control':trace['same_count_random']},common,ready,explanatory=True)
        contexts[seed] = parent,limits,cost,common,pairs
        del q0, chosen, random

    bounds = route_bounds(records,load_rules())
    atomic_json(output/'route_bounds.json',bounds)
    retest_records = [{'seed':r['seed'], 'method':study.SELECTED if r['method']==CANDIDATE else r['method'],
                      'panels':{p['panel_id']:p['blocks'] for p in r['heldout']}} for r in records if r['method'] != HAD]
    placements = {p['panel_id']:p['placement'] for p in records[0]['heldout']}
    paired, _ = study.assess_documents(retest_records,config['seeds'],placements)
    atomic_json(output/'paired_parent_and_random.json',paired)
    stamp = {'pilot_variant':CANDIDATE,'route_bounds':bounds,'source_type':'real_layer','all_controls_requested':all_controls}
    if bounds['both_routes_ruled_out'] and not all_controls:
        return {**stamp,'status':'RETEST_STOPPED_BY_BOUND','next_step':'stop_batch_variant','complete_comparison':False}

    for seed in config['seeds']:
        ready = prepared['seeds'][seed]
        parent,limits,cost,common,pairs = contexts[seed]
        for method in CONTROLS:
            report(progress,'retest.matched_control',seed=seed,method=method)
            count = design['matching_grid_points']
            result = fit_baseline(method,ready['fit'],parent,ready['specs'],pairs,remaining(cfg,cost),budgets=limits,
                                  grid_points=count,ste_steps=count-1,progress=progress)
            q = result.pop('q')
            if method in ('angle_grid_matched','givens_ste_matched') and any(r['evaluations']!=count for r in result['log']):
                result.update(status='RESOURCE_BLOCKED',reason='Incomplete per-pair proposal budget')
            marginal = result['seconds']
            save(seed,method,q,cost+marginal,{**result,'parent_seconds':cost,'control_seconds':marginal},common,ready)
            del q
    summary = summarize_pilot_results(paths)
    if engineering_pairs is not None:
        return {**stamp,**summary,'scope':'pretrained engineering verification','engineering_pairs':engineering_pairs,
                'raw_results':[str(p) for p in paths]}
    gate = build_g1_evidence(paths,output,g0_evidence_path=config['g0_evidence_path'],batch_retest=True)
    if summary['status']=='COMPLETE':
        atomic_json(output/'document_intervals.json',analyze_run(output))
    return {**stamp,**summary,'complete_comparison':summary['status']=='COMPLETE','gate':gate['status'],
            'raw_results':[str(p) for p in paths]}
