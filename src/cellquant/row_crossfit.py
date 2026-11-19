"""Swap only the first up projection's calibration rows in a matched shared-grid fit."""
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch

from .batch_retest import fit_parent
from .bilateral_rows import PRECISIONS, SITE, row_energies
from .bilateral_screen import prefix_rotation, replay_retest_parent
from .matching_batches import load_matching_arms, q_digest, replay_continued_grid
from .matching_study import CONTINUED, WARM, paired_site_gains
from .pilot import panel_metrics, pilot_solver_config
from .pilot_bootstrap import document_resamples, document_statistics
from .progress import report
from .runtime import atomic_json, config_hash, sha256_file
from .solver import pair_schedule, panel_fingerprint
from .tensor_solver import hadamard_rotation, synchronize


COHORT_FILE = 'row_crossfit_cohort_7b.json'
DESIGN = {'version':1, 'seeds':[11,23], 'site':SITE, 'rows_per_fit':64,
          'partition_namespace':2026111906, 'row_seed':2026111906, 'bootstrap_seed':2026111906, 'bootstrap_draws':10000,
          'documents_per_seed':16, 'tokens_per_document':32, 'grid_pairs':256, 'grid_points':129,
          'primary':'B-fit gain vs A-fit on B minus the same gain on A, W4A4, percentage points',
          'transfer':'B-fit gain vs A-fit on all rows outside both fits and on the full projection'}
GROUPS = ('sampled','alternate','complement','all_rows')


def alternate_row_ids(total, original, row_seed):
    """Draw one non-fit row per contiguous stratum without using model weights or scores."""
    original = np.asarray(original)
    if (original.ndim != 1 or len(original) == 0 or original.dtype.kind not in 'iu' or
            2*len(original) >= total or np.unique(original).size != len(original) or original.min() < 0 or original.max() >= total):
        raise ValueError('Invalid original fit rows')
    rng = np.random.default_rng(row_seed)
    rows = []
    for stratum in np.array_split(np.arange(total), len(original)):
        available = np.setdiff1d(stratum, original, assume_unique=True)
        if not len(available):
            raise ValueError('A fitted stratum has no alternative row')
        rows.append(int(rng.choice(available)))
    return rows


def paired_row_effect(by_seed, new, baseline, groups, precision, *, draws, rng_seed, difference=None, paired_orders=()):
    """Use one document multiplicity across methods and row groups, with the fitted seeds fixed."""
    records = []
    seeds = sorted(by_seed)
    pairs = [(new, baseline), *paired_orders]
    methods = list(dict.fromkeys(method for pair in pairs for method in pair))
    for seed in seeds:
        for method in methods:
            data = by_seed[seed]['methods'][method]['blocks']
            panels = {g:[dict(block, squared_error=block[precision]) for block in data[g]] for g in groups}
            records.append({'seed':seed,'method':method,'panels':panels})
    stats, strata, docs = document_statistics(records, seeds, groups, methods)
    indices = [(methods.index(candidate), methods.index(control)) for candidate, control in pairs]
    weights = document_resamples(strata, len(docs), draws, rng_seed)
    weights = np.vstack((np.ones(len(docs), dtype=int), weights))
    gains = []
    for start in range(0,len(weights),256):
        totals = np.einsum('bd,msldk->bmslk', weights[start:start+256], stats, optimize=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            nmse = totals[...,0] / totals[...,1]
            # Keep each continuation paired with its own starting grid in every document draw.
            gains.append(np.mean([100*(1-nmse[:,i]/nmse[:,j]) for i, j in indices], axis=0))
    gains = np.concatenate(gains)

    def interval(values):
        if not np.isfinite(values).all():
            return {'undefined':'Zero target or comparator energy in a paired resample'}
        mean = values.mean(axis=1)
        return {'estimate':float(mean[0]), 'paired_95':np.percentile(mean[1:], [2.5,97.5]).tolist(),
                'by_seed':dict(zip(map(str,seeds),values[0].tolist()))}

    result = {g:interval(gains[:,:,i]) for i,g in enumerate(groups)}
    if difference is not None:
        left, right = (groups.index(g) for g in difference)
        result['difference_pp'] = interval(gains[:,:,left] - gains[:,:,right])
    return result


def summarize_crossfit(by_seed, design):
    options = {'draws':design['bootstrap_draws'], 'rng_seed':design['bootstrap_seed']}
    contrasts = {p:paired_row_effect(by_seed, 'grid_b', 'grid_a', GROUPS, p, difference=('alternate','sampled'), **options)
                 for p in PRECISIONS}
    trajectories = {arm:paired_row_effect(by_seed, 'grid_'+arm, 'prefix_'+arm, GROUPS, 'W4A4', **options)
                    for arm in ('a','b')}
    primary = contrasts['W4A4']
    effect = primary['difference_pp']
    replicated = 'undefined' not in effect and effect['paired_95'][0] > 0 and all(
        primary['sampled']['by_seed'][s] < 0 < primary['alternate']['by_seed'][s] for s in effect['by_seed'])
    return {'grid_b_vs_grid_a':contrasts, 'full_vs_short_within_fit':trajectories, 'interaction_replicated':replicated,
            'interpretation':'both full grids prefer their own rows in each seed and the paired interaction excludes zero'
                             if replicated else 'the prespecified reciprocal row advantage was not established'}


@torch.inference_mode()
def run_crossfit(config, prepared, weight, output, reference, *, fit_7b=None, progress=None):
    design = config['row_crossfit']['design']
    dimension = prepared['hidden_size']
    if design != {**DESIGN, 'grid_pairs':32 if fit_7b is None else 256} or config['seeds'] != design['seeds']:
        raise ValueError('Unexpected cross-fit protocol')
    cfg = pilot_solver_config(config)
    short = design['grid_pairs']//4
    training = {'prefix_pairs':short*2, 'parent_grid_points':design['grid_points']}
    grid_refs = load_matching_arms(reference, config['seeds']) if fit_7b is None else None
    raw, by_seed, other_sites = {}, {}, []
    for seed in config['seeds']:
        ready = prepared['seeds'][seed]
        fit, dev, specs = ready['fit'], ready['heldout'], ready['specs']
        schedule = pair_schedule(dimension, design['grid_pairs'], seed)
        if fit_7b is None:
            ref = grid_refs[seed][CONTINUED]
            expected_hash = ref['fit_panel_sha256']
            stages = [grid_refs[seed][WARM]['solver'],ref['solver']]
            grid = replay_continued_grid(grid_refs[seed], dimension, seed, cfg.device)
        else:
            ref = fit_7b['grids'][seed]
            expected_hash = ref['fit_panel_fingerprint']
            stages = ref['solver']['grid_stages']
            grid = replay_retest_parent(ref,cfg.device)
        if (panel_fingerprint(fit) != expected_hash or len(fit) != 9 or len(dev) != 9
                or len(stages) != 2 or any(s['status'] != 'COMPLETE' or s['completed_pairs'] != 2*short for s in stages)
                or [list(pair) for pair in schedule] != [v['pair'] for stage in stages for v in stage['log']]
                or any(v['evaluations'] != design['grid_points'] for s in stages for v in s['log'])):
            raise ValueError('Original calibration panels or grid schedule differ from the archived fit')
        target = next(p for p in fit if p.panel_id == SITE)
        held = next(p for p in dev if p.panel_id == SITE)
        fit_docs, dev_docs = target.metadata['document_ids'], held.metadata['document_ids']
        original = target.metadata['weight_row_ids']
        if (original != np.linspace(0,len(weight)-1,design['rows_per_fit'],dtype=int).tolist()
                or len(fit_docs) != 8 or len(set(fit_docs)) != 8 or len(set(dev_docs)) != design['documents_per_seed']
                or sorted(fit_docs) != ref['fit_documents'] or set(fit_docs).intersection(dev_docs)
                or original != held.metadata['weight_row_ids'] or any(p.metadata['source_type'] != 'real_layer' for p in fit+dev)
                or any(p.metadata['document_ids'] != dev_docs or p.metadata['tokens_per_block'] != [design['tokens_per_document']]*len(dev_docs) for p in dev)
                or (fit_7b is not None and any(p.metadata['graph_identity'] != ref['graph_identity'] for p in fit+dev))):
            raise ValueError('Unexpected fit or held-out data for paired row comparison')
        alternate = alternate_row_ids(len(weight),original,design['row_seed'])
        prefix_a, prefix_length = prefix_rotation(dimension, seed, stages[0], cfg.device)
        if prefix_length != short:
            raise ValueError('Unexpected archived short grid')
        report(progress,'row_crossfit.fit',seed=seed,pairs=len(schedule))
        synchronize(cfg.device)
        start = time.perf_counter()
        swapped = replace(target, weight=weight[alternate].float(), metadata={**target.metadata,'weight_row_ids':alternate})
        fit_b = [swapped if p.panel_id == SITE else p for p in fit]
        initial = hadamard_rotation(dimension,seed,cfg.device)
        synchronize(cfg.device)
        cost = ready['capture_seconds'] + time.perf_counter()-start
        grid_b, limits, seconds, solved = fit_parent(fit_b,initial,specs,schedule,cfg,training,cost,progress)
        prefix_b, _ = prefix_rotation(dimension, seed, solved[0], cfg.device)
        rotations = {'grid_a':grid,'prefix_a':prefix_a,'grid_b':grid_b,'prefix_b':prefix_b}
        methods = {}
        for method, q in rotations.items():
            report(progress,'row_crossfit.score',seed=seed,method=method)
            rows = row_energies(weight,held,q,specs,alternate_rows=alternate)
            methods[method] = {**rows,'q_sha256':q_digest(q),
                               'orthogonality_residual':float(torch.linalg.norm(q.T@q-torch.eye(dimension,dtype=q.dtype,device=q.device))/dimension**.5)}
            if method in ('grid_a','grid_b'):
                measures = panel_metrics([p for p in dev if p.panel_id != SITE],q,specs,cfg.device)
                methods[method]['other_panels'] = measures
                other_sites.append({'seed':seed,'method':method,'panels':{m['panel_id']:m['blocks'] for m in measures}})
        row = {'seed':seed,'source_type':'real_layer', 'rows_a':original, 'rows_b':alternate,'methods':methods, 'site':SITE,
               'fit_documents':fit_docs,'heldout_documents':dev_docs,'fit_a_panel_sha256':expected_hash,
               'fit_b_panel_sha256':panel_fingerprint(fit_b),'heldout_panel_sha256':panel_fingerprint(dev),
               'grid_a_reference_sha256':sha256_file(reference),'grid_a_limits':stages[0]['limits'],
               'grid_b_stages':solved,'grid_b_limits':limits,
               'pair_schedule_sha256':config_hash(schedule),'grid_b_calibration_seconds':seconds,'capture_seconds':ready['capture_seconds']}
        path = atomic_json(Path(output)/f'seed_{seed}'/'row_crossfit.json',row)
        raw[path.relative_to(output).as_posix()] = sha256_file(path)
        by_seed[seed] = row
        del grid, grid_b, prefix_a, prefix_b, initial, rotations, q
    placements = {p.panel_id:p.placement for p in dev if p.panel_id != SITE}
    other = paired_site_gains(other_sites,config['seeds'],placements,{'grid_b_vs_grid_a':('grid_b','grid_a')},
                              repetitions=design['bootstrap_draws'],rng_seed=design['bootstrap_seed'])
    return {'source_type':'real_layer','evidence_kind':'MATCHED_ROW_CROSSFIT','status':'STUDY_COMPLETE','design':design,
            'reference_archive_sha256':sha256_file(reference),'row_artifacts':raw,'crossfit_summary':summarize_crossfit(by_seed,design),
            'other_eight_sites':other,'timing_scope':'A fit is replayed; only the B fit was timed in this invocation'}
