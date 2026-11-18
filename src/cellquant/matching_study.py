"""Evaluate output-side matching beyond a fitted shared grid rotation on fresh documents."""
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

from .events import ResourceLimit
from .output_matching import disjoint_pairs, fit_matching
from .pilot import panel_metrics, pilot_solver_config
from .pilot_bootstrap import document_resamples, document_statistics
from .progress import report
from .runtime import atomic_json, config_hash, read_json, sha256_file
from .solver import pair_schedule, panel_fingerprint
from .tensor_solver import fit_baseline, grid_angles, hadamard_rotation, synchronize


HAD = 'hadamard_shared'
WARM, CONTINUED = 'warm_grid', 'continued_grid'
SPARSE, WEIGHT, PRODUCT, UNFILTERED = 'product_prefix', 'weight_full', 'product_full', 'product_joint_check_only'
HYBRID = 'continued_grid_plus_matching'
KIND = 'OUTPUT_MATCHING_STUDY'
DESIGN = {'version':3, 'partition_namespace':2026111802, 'warm_pairs':128, 'extra_grid_pairs':128,
          'refine_continued_grid':True,
          'sparse_pairs':128, 'baseline_grid_points':129, 'matching_grid_points':17,
          'step_scales':[0., .25, .5, 1.], 'bootstrap_draws':10000, 'bootstrap_seed':20261118,
          'primary':'product_full vs warm_grid, median of three output sites'}


def paired_site_gains(records, seeds, placements, contrasts, *, repetitions=10000, rng_seed=20261118):
    """Paired document bootstrap. Hold sites, completed fits and calibration seeds fixed."""
    sites = sorted(placements)
    methods = sorted({r['method'] for r in records})
    stats, strata, docs = document_statistics(records, seeds, sites, methods)
    weights = document_resamples(strata, len(docs), repetitions, rng_seed)
    weights = np.vstack((np.ones(len(docs), dtype=int), weights))
    groups = {'all':np.ones(len(sites), dtype=bool)}
    for side in ('input','output'):
        mask = np.array([placements[s] == side for s in sites])
        if mask.any():
            groups[side] = mask
    contrasts_out = {}
    for label, (new, ref) in contrasts.items():
        pair = stats[[methods.index(new), methods.index(ref)]]
        samples = {g:[] for g in groups}
        point = None
        for i in range(0, len(weights), 256):
            summed = np.einsum('bd,msldk->bmslk',weights[i:i+256],pair,optimize=True)
            if (summed[...,1] <= 0).any() or (summed[:,1,...,0] <= 0).any():
                break
            nmse = summed[...,0] / summed[...,1]
            gain = 100 * (1 - nmse[:,0] / nmse[:,1])
            if i == 0:
                point = gain[0]
            for group, mask in groups.items():
                samples[group].append(np.median(gain.mean(axis=1)[:,mask], axis=1))
        else:
            item = {'site_gain_pct':dict(zip(sites,point.mean(axis=0).tolist()))}
            for group, mask in groups.items():
                draws = np.concatenate(samples[group])
                item[group] = {'median_gain_pct':float(draws[0]),
                               'paired_95_pct':np.percentile(draws[1:], [2.5,97.5]).tolist(),
                               'seed_medians_pct':dict(zip(map(str,seeds), np.median(point[:,mask],axis=1).tolist()))}
            contrasts_out[label] = item
            continue
        contrasts_out[label] = {'undefined':'Zero target energy or comparator NMSE in a document resample'}
    return {'seeds_held_fixed':list(seeds),'sites':sites,'distinct_documents':len(docs),
            'bootstrap_draws':repetitions,'bootstrap_seed':rng_seed,'comparisons':contrasts_out}


def attribution(panels):
    out = {}
    for p in panels:
        blocks = p['blocks']
        energy = sum(b['target_energy'] for b in blocks)
        if energy <= 0:
            raise ValueError('Cannot attribute error on a zero-energy held-out site')
        parts = {key:sum(b[key] for b in blocks)/energy
                 for key in ('weight_error_energy','activation_error_energy','cross_energy')}
        parts['closure_abs_error'] = abs(sum(parts.values())-p['nmse'])
        out[p['panel_id']] = parts
    return out


def explain_matching(rows, parent=WARM, method=PRODUCT):
    """Additive changes in units of the warm held-out error, separately per seed/site."""
    result = {}
    for seed in sorted({r['seed'] for r in rows}):
        arms = {r['method']:r for r in rows if r['seed']==seed}
        warm, new = arms[parent], arms[method]
        base_nmse = {p['panel_id']:p['nmse'] for p in warm['heldout']}
        explanations = {}
        for p in new['heldout']:
            site = p['panel_id']
            h = base_nmse[site]
            if h <= 0:
                explanations[site] = {'undefined':'Zero warm NMSE'}
                continue
            before, after = warm['heldout_error_terms'][site], new['heldout_error_terms'][site]
            parts = {k:100*(before[k]-after[k])/h for k in ('weight_error_energy','activation_error_energy','cross_energy')}
            observed = 100*(1-p['nmse']/h)
            explanations[site] = {'placement':p['placement'],'gain_pct':observed,'contributions_pp':parts,
                                  'closure_gap_pp':observed-sum(parts.values())}
        result[str(seed)] = {'sites':explanations, 'selected_scale':new['solver']['selected_scale'],
            'applied_pairs':new['solver']['nonzero_applied_angles'], 'joint_fit_trials':new['solver']['joint_fit_trials'],
            'calibration_seconds':{m:r['total_calibration_seconds'] for m,r in arms.items()}}
    return result


@torch.inference_mode()
def run_matching_seed(ready, q0, seed, config, design, output, fingerprint, progress=None):
    fit, dev, specs = ready['fit'], ready['heldout'], ready['specs']
    cfg = pilot_solver_config(config, execution_fingerprint=fingerprint)
    fit_docs = set(fit[0].metadata['document_ids'])
    dev_docs = set(dev[0].metadata['document_ids'])
    placements = {p.panel_id:p.placement for p in fit}
    if (fit_docs & dev_docs or not fit_docs or not dev_docs
            or {p.panel_id:p.placement for p in dev} != placements
            or any(set(p.metadata['document_ids']) != fit_docs for p in fit)
            or any(set(p.metadata['document_ids']) != dev_docs for p in dev)):
        raise ValueError('Fit and held-out document roles or panel identities disagree')
    sources = {p.metadata['source_type'] for p in [*fit,*dev]}
    if len(sources) != 1 or not sources <= {'unit_test_fixture','real_layer'}:
        raise ValueError('Inconsistent matching study provenance')
    source = sources.pop()
    if source == 'real_layer' and (len(fit_docs) != 8 or len(dev_docs) != 8 or len(placements) != 9):
        raise ValueError('The real matching study requires eight documents per split and all nine sites')
    prefix = design['warm_pairs']
    schedule = pair_schedule(len(q0), prefix + design['extra_grid_pairs'], seed)
    pairs = disjoint_pairs(len(q0), seed)
    fit_sha = panel_fingerprint(fit)
    heldout_sha = panel_fingerprint(dev)
    refs, records = [], []

    def save(method, q, solver, calibration_seconds):
        report(progress, 'matching.evaluate', seed=seed, method=method)
        metrics = panel_metrics(dev, q, specs, cfg.device, error_terms=True)
        orth = float(torch.linalg.norm(q.T@q - torch.eye(len(q), device=q.device, dtype=q.dtype))/math.sqrt(len(q)))
        row = {'source_type':source, 'evidence_kind':KIND, 'seed':seed, 'method':method, 'solver':solver,
               'orientation':'R1=Q.T','q_sha256':hashlib.sha256(q.contiguous().cpu().numpy().tobytes()).hexdigest(),
               'orthogonality_residual':orth,'fit_panel_sha256':fit_sha,'heldout_panel_sha256':heldout_sha,
               'fit_documents':sorted(fit_docs),'heldout_documents':sorted(dev_docs),'heldout':metrics,
               'heldout_error_terms':attribution(metrics),'total_calibration_seconds':calibration_seconds}
        path = atomic_json(Path(output)/f'seed_{seed}'/method/'matching_result.json', row)
        refs.append({'path':str(path.resolve()), 'sha256':sha256_file(path)})
        records.append({'seed':seed,'method':method,'panels':{p['panel_id']:p['blocks'] for p in metrics}})

    def grid(method, start_q, axes, limits=None):
        def emit(phase, **details):
            report(progress,'matching.'+phase,seed=seed,method=method,**details)
        solved = fit_baseline('angle_grid_matched',fit,start_q,specs,axes,cfg,
                              grid_points=design['baseline_grid_points'],progress=emit,budgets=limits)
        q = solved.pop('q')
        if solved['status'] != 'COMPLETE':
            atomic_json(Path(output)/f'seed_{seed}'/method/'incomplete_solver.json',solved)
            raise ResourceLimit(solved.get('reason') or 'Incomplete grid control')
        # A timed-out angular scan can complete a pair after evaluating only part of its grid.
        if any(r['evaluations'] != len(grid_angles_used) for r in solved['log']):
            atomic_json(Path(output)/f'seed_{seed}'/method/'incomplete_solver.json',solved)
            raise ResourceLimit('A matched grid step evaluated fewer than its full candidate set')
        return q, solved

    grid_angles_used = grid_angles(q0,cfg.theta_limit,design['baseline_grid_points'])
    capture = ready['capture_seconds'] + ready['hadamard_seconds']
    warm, first = grid(WARM,q0,schedule[:prefix])
    limits = first['limits']
    warm_seconds = capture + first['seconds']
    save(HAD, q0, {'seed':seed, 'factors':[], 'fit_metrics':first['initial']}, capture)
    save(WARM, warm, first, warm_seconds)
    # Fit choices below only read fit tensors. All registered arms run regardless of earlier scores.
    continued, extra = grid(CONTINUED,warm,schedule[prefix:],limits)
    continued_seconds = warm_seconds+extra['seconds']
    save(CONTINUED,continued,{**extra,'parent':WARM},continued_seconds)
    arms = [(SPARSE,'product',pairs[:design['sparse_pairs']]), (WEIGHT,'weight',pairs),
            (UNFILTERED,'product',pairs),(PRODUCT,'product',pairs)]
    def measure_matching(method, criterion, axes, parent, start_q, prior_seconds):
        def emit(phase, **details):
            report(progress,phase,seed=seed,method=method,**details)
        q, info = fit_matching(fit,start_q,specs,axes,cfg,limits,criterion=criterion,
                               grid_points=design['matching_grid_points'],step_scales=design['step_scales'],
                               allocate_risk=method != UNFILTERED,progress=emit)
        save(method,q,{**info,'parent':parent},prior_seconds+info['seconds'])
    for method, criterion, axes in (arms if seed == config['seeds'][0] else reversed(arms)):
        measure_matching(method,criterion,axes,WARM,warm,warm_seconds)
    if design.get('refine_continued_grid'):
        measure_matching(HYBRID,'product',pairs,CONTINUED,continued,continued_seconds)
    return records, refs, placements, source


def run_matching_study(config, prepared, output, *, execution_fingerprint, progress=None):
    design = config['output_matching_study']['design']
    records, refs, sources = [], [], set()
    site_sets = []
    for seed in config['seeds']:
        ready = prepared['seeds'].pop(seed)
        device = config['solver']['device']
        synchronize(device)
        started = time.monotonic()
        q0 = hadamard_rotation(prepared['hidden_size'],seed,device)
        synchronize(device)
        ready['hadamard_seconds'] = time.monotonic()-started
        rows, paths, placements, source = run_matching_seed(ready,q0,seed,config,design,output,execution_fingerprint,progress)
        records.extend(rows)
        refs.extend(paths)
        sources.add(source)
        site_sets.append(placements)
        del q0, ready
    if len(sources) != 1 or any(p != site_sets[0] for p in site_sets):
        raise ValueError('Inconsistent provenance or sites across seeds')
    report(progress,'matching.paired_documents')
    stats = paired_site_gains(records,config['seeds'],site_sets[0],study_contrasts(design),
                              repetitions=design['bootstrap_draws'],rng_seed=design['bootstrap_seed'])
    raw = [read_json(ref['path']) for ref in refs]
    stats['mechanism_by_seed'] = explain_matching(raw)
    if design.get('refine_continued_grid'):
        stats['hybrid_mechanism_by_seed'] = explain_matching(raw,CONTINUED,HYBRID)
    result = {'status':'STUDY_COMPLETE','source_type':sources.pop(),'evidence_kind':KIND,
              'design':design,'design_sha256':config_hash(design),'matching_artifacts':refs,'matching_summary':stats}
    atomic_json(Path(output)/'matching_summary.json',result)
    return result


def study_contrasts(design):
    contrasts = {'coverage':(PRODUCT,SPARSE), 'activation_information':(PRODUCT,WEIGHT),
                 'primary_vs_warm_grid':(PRODUCT,WARM),'vs_continued_grid':(PRODUCT,CONTINUED),
                 'vs_hadamard':(PRODUCT,HAD),'risk_allocation':(PRODUCT,UNFILTERED)}
    if design.get('refine_continued_grid'):
        contrasts['hybrid_vs_continued_grid'] = (HYBRID,CONTINUED)
    return contrasts
