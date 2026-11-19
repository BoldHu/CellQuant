"""Continue a shared rotation on different output rows at a fixed grid-search budget."""
from dataclasses import replace
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from .batch_retest import remaining
from .bilateral_documents import checked_json
from .bilateral_rows import SITE, row_energies
from .events import ResourceLimit
from .matching_batches import load_matching_arms, q_digest
from .matching_study import CONTINUED, WARM, paired_site_gains
from .output_matching import fits_limits
from .pilot import panel_metrics, pilot_solver_config
from .progress import report
from .row_crossfit import DESIGN as CROSSFIT, GROUPS, alternate_row_ids, paired_row_effect
from .runtime import atomic_json, config_hash, sha256_file
from .solver import pair_schedule, panel_fingerprint
from .tensor_solver import fit_baseline, hadamard_rotation, rotate_rows


COHORT_FILE = 'row_refresh_cohort_7b.json'
REFERENCE_FILE = 'row_crossfit_7b.zip'
DESIGN = {**CROSSFIT, 'partition_namespace':2026111907, 'bootstrap_seed':2026111907,
          'methods':['aa','ab','bb','ba'], 'switch_fraction':0.5,
          'primary':'mean of AB vs AA and BA vs BB W4A4 gains on rows outside A and B',
          'transfer':'whole first up projection and each of the other eight sites',
          'limits':'second-stage limits of the destination panel at the original Hadamard',
          'advance':'primary lower 95% bound and both seed gains positive, whole projection gain positive, both crossed fits finish within destination limits'}
ORDERS = {'ab_vs_aa':('ab','aa'), 'ba_vs_bb':('ba','bb')}


def read_refresh_archive(path, crossfit, grid, *, small):
    """Read the four measured rotations and their document energies, bound to both parent archives."""
    prefix = '' if small else 'run/'
    with ZipFile(path) as archive:
        result = json.loads(archive.read(prefix+'result.json'))
        design = {**DESIGN,'grid_pairs':32 if small else 256}
        if (result['status'] != 'STUDY_COMPLETE' or result['source_type'] != 'real_layer'
                or result['evidence_kind'] != 'MATCHED_ROW_REFRESH' or result['design'] != design
                or (not small and result.get('execution_mode') != 'matched_row_refresh')
                or result['reference_archive_sha256'] != sha256_file(grid) or result['crossfit_reference_sha256'] != sha256_file(crossfit)):
            raise ValueError('Row-refresh archive differs from the fitted models or study design')
        rows = {}
        for seed in design['seeds']:
            name = f'seed_{seed}/row_refresh.json'
            row = checked_json(archive,prefix+name,result['row_artifacts'][name])
            if (row['seed'] != seed or row['source_type'] != 'real_layer' or row['site'] != SITE
                    or set(row['methods']) != set(design['methods']) or set(row['tails']) != {'ab','ba'}):
                raise ValueError('Incomplete row-refresh measurements')
            rows[seed] = row
    return result, rows


def read_crossfit_archive(path, grid_archive, *, small):
    """Load the completed row comparison and bind its raw measurements to their original fits."""
    prefix = '' if small else 'run/'
    with ZipFile(path) as archive:
        result = json.loads(archive.read(prefix+'result.json'))
        design = {**CROSSFIT, 'grid_pairs':32 if small else 256}
        if (result['status'] != 'STUDY_COMPLETE' or result['source_type'] != 'real_layer'
                or result['evidence_kind'] != 'MATCHED_ROW_CROSSFIT' or result['design'] != design
                or result['reference_archive_sha256'] != sha256_file(grid_archive)):
            raise ValueError('Expected the completed row cross-fit with the same grid and design')
        if not small and result.get('execution_mode') != 'matched_row_crossfit':
            raise ValueError('Expected the A100 row-crossfit runner')
        rows = {}
        for seed in design['seeds']:
            name = f'seed_{seed}/row_crossfit.json'
            row = checked_json(archive,prefix+name,result['row_artifacts'][name])
            size = 1376 if small else 11008
            original = np.linspace(0,size-1,64,dtype=int).tolist()
            if (row['source_type'] != 'real_layer' or row['seed'] != seed or row['site'] != SITE
                    or row['rows_a'] != original or row['rows_b'] != alternate_row_ids(size,original,design['row_seed'])
                    or set(row['methods']) != {'grid_a','grid_b','prefix_a','prefix_b'}
                    or row['grid_a_reference_sha256'] != result['reference_archive_sha256']):
                raise ValueError('Incompatible row calibration or measurement identity')
            rows[seed] = row
    return result, rows


def replay_grids(stages, hidden, seed, device, arm, source, pairs):
    """Rebuild both checkpoints from exactly the accepted steps of the recorded grid."""
    q = hadamard_rotation(hidden,seed,device)
    half = len(pairs)//2
    if len(stages) != 2 or len(pairs) % 4:
        raise ValueError('Expected two equal grid segments')
    for i, stage in enumerate(stages):
        log = stage['log']
        expected = [list(p) for p in pairs[i*half:(i+1)*half]]
        if (stage['status'] != 'COMPLETE' or stage['completed_pairs'] != half or not stage['shared_global_r1']
                or [v['pair'] for v in log] != expected or [v['pair_index'] for v in log] != list(range(half))
                or any(v['evaluations'] != DESIGN['grid_points'] for v in log) or stage['limits'] != source['grid_'+arm+'_limits']):
            raise ValueError('Reference grid schedule, evaluations or scope differ')
        for j, step in enumerate(log):
            if step['accepted']:
                rotate_rows(q,step['pair'],step['best']['theta'])
            if i == 0 and j+1 == half//2 and q_digest(q) != source['methods']['prefix_'+arm]['q_sha256']:
                raise ValueError('Short grid replay differs from the prior real measurements')
        if i == 0:
            middle = q.clone()
    if q_digest(q) != source['methods']['grid_'+arm]['q_sha256']:
        raise ValueError('Full grid replay differs from the prior real measurements')
    return middle, q


def summarize_refresh(rows, design):
    options = {'draws':design['bootstrap_draws'], 'rng_seed':design['bootstrap_seed']}
    effects = {name:paired_row_effect(rows,*pair,GROUPS,'W4A4',**options) for name,pair in ORDERS.items()}
    mean = paired_row_effect(rows,'ab','aa',GROUPS,'W4A4',paired_orders=[('ba','bb')],**options)
    primary = mean['complement']
    feasible = all(record['tails'][arm]['destination_feasible_at_end'] for record in rows.values() for arm in ('ab','ba'))
    if 'undefined' in primary or 'undefined' in mean['all_rows']:
        decision = 'insufficient_evidence'
    elif (feasible and primary['paired_95'][0] > 0 and min(primary['by_seed'].values()) > 0
          and mean['all_rows']['estimate'] > 0):
        decision = 'replicate_refresh_on_more_projections'
    else:
        decision = 'retain_row_effect_stop_single_switch'
    return {'by_order':effects,'mean_paired_orders':mean,'both_orders_feasible':feasible,'next_step':decision}


@torch.inference_mode()
def run_refresh(config, prepared, weight, output, reference, crossfit_archive, *, fit_7b=None, progress=None):
    small = fit_7b is None
    design = config['row_refresh']['design']
    if design != {**DESIGN,'grid_pairs':32 if small else 256} or config['seeds'] != design['seeds']:
        raise ValueError('Row-refresh protocol differs from its registered study')
    previous, originals = read_crossfit_archive(crossfit_archive,reference,small=small)
    if not small and previous['crossfit_summary']['interaction_replicated'] is not True:
        raise ValueError('The parent 7B row effect was not replicated')
    grid_refs = load_matching_arms(reference,design['seeds']) if small else None
    cfg = pilot_solver_config(config)
    hidden = prepared['hidden_size']
    fit_rows, raw, other = {}, {}, []
    for seed in design['seeds']:
        source = originals[seed]
        ready = prepared['seeds'][seed]
        fit, dev, specs = ready['fit'],ready['heldout'],ready['specs']
        target = next(p for p in fit if p.panel_id == SITE)
        held = next(p for p in dev if p.panel_id == SITE)
        rows_b = source['rows_b']
        changed = replace(target,weight=weight[rows_b].float(),metadata={**target.metadata,'weight_row_ids':rows_b})
        panels = {'a':fit,'b':[changed if p.panel_id == SITE else p for p in fit]}
        docs = held.metadata['document_ids']
        if (len(fit) != 9 or len(dev) != 9 or any(p.metadata['source_type'] != 'real_layer' for p in fit+dev)
                or set(target.metadata['document_ids']) != set(source['fit_documents'])
                or held.metadata['weight_row_ids'] != target.metadata['weight_row_ids'] or target.metadata['weight_row_ids'] != source['rows_a']
                or len(docs) != design['documents_per_seed'] or len(set(docs)) != len(docs)
                or set(docs).intersection(source['fit_documents'] + source['heldout_documents'])
                or any(p.metadata['document_ids'] != docs or p.metadata['tokens_per_block'] != [design['tokens_per_document']]*len(docs) for p in dev)
                or any(panel_fingerprint(ps) != source['fit_'+arm+'_panel_sha256'] for arm,ps in panels.items())):
            raise ValueError('Unexpected fit rows, articles, or quantization panels')
        if small:
            a = grid_refs[seed]
            stages_a = [a[WARM]['solver'],a[CONTINUED]['solver']]
            prefix_a_seconds = a[WARM]['total_calibration_seconds']
        else:
            a = fit_7b['grids'][seed]
            if any(p.metadata['graph_identity'] != a['graph_identity'] for p in fit+dev):
                raise ValueError('The captured graph differs from the original full grid')
            stages_a = a['solver']['grid_stages']
            prefix_a_seconds = a['total_calibration_seconds'] - stages_a[1]['seconds']
        stages = {'a':stages_a, 'b':source['grid_b_stages']}
        prefix_seconds = {'a':prefix_a_seconds,'b':source['grid_b_calibration_seconds']-stages['b'][1]['seconds']}
        schedule = pair_schedule(hidden,design['grid_pairs'],seed)
        if config_hash(schedule) != source['pair_schedule_sha256']:
            raise ValueError('Unexpected row-refresh coordinate schedule')
        scales = {arm:source['grid_'+arm+'_limits'] for arm in panels}
        methods, tails = {}, {}
        for origin,destination in (('a','b'),('b','a')):
            fixed, crossed = origin+origin, origin+destination
            start, control = replay_grids(stages[origin],hidden,seed,cfg.device,origin,source,schedule)
            # The input preparation and first segment belong to this arm even though they were cached.
            basis_cost = prefix_seconds[origin]
            report(progress,'row_refresh.fit',seed=seed,order=crossed,remaining_pairs=len(schedule)//2)
            tail = fit_baseline('angle_grid_matched',panels[destination],start,specs,schedule[len(schedule)//2:],
                                remaining(cfg,basis_cost),grid_points=design['grid_points'],budgets=scales[destination],progress=progress)
            updated = tail.pop('q')
            accounted = basis_cost + tail['seconds']
            if (tail['status'] != 'COMPLETE' or tail['completed_pairs'] != len(schedule)//2 or accounted > cfg.wall_seconds
                    or any(t['evaluations'] != design['grid_points'] for t in tail['log'])):
                raise ResourceLimit(f'Row refresh {crossed} failed to complete its matched calibration budget')
            tails[crossed] = {'initial_q_sha256':q_digest(start),'solver':tail,'cached_prefix_seconds':basis_cost,
                             'accounted_calibration_seconds':accounted,'fixed_calibration_seconds':basis_cost+stages[origin][1]['seconds'],
                             'destination_feasible_at_start':fits_limits(tail['initial'],scales[destination],cfg.budget_tolerance),
                             'destination_feasible_at_end':fits_limits(tail['final'],scales[destination],cfg.budget_tolerance)}
            for label,q in ((fixed,control),(crossed,updated)):
                report(progress,'row_refresh.score',seed=seed,order=label)
                measurements = panel_metrics([p for p in dev if p.panel_id != SITE],q,specs,cfg.device)
                methods[label] = {**row_energies(weight,held,q,specs,alternate_rows=rows_b),'q_sha256':q_digest(q),
                                  'orthogonality_residual':float(torch.linalg.norm(q.T@q-torch.eye(hidden,device=q.device,dtype=q.dtype))/hidden**.5),
                                  'other_panels':measurements}
                other.append({'seed':seed,'method':label,'panels':{m['panel_id']:m['blocks'] for m in measurements}})
            del q, start, control, updated
        row = {'seed':seed,'source_type':'real_layer','site':SITE,'methods':methods,'tails':tails,'rows_a':source['rows_a'],'rows_b':rows_b,
               'fit_documents':source['fit_documents'],'heldout_documents':docs,'heldout_panel_sha256':panel_fingerprint(dev),
               'fit_panel_hashes':{arm:source['fit_'+arm+'_panel_sha256'] for arm in panels},'pair_schedule_sha256':config_hash(schedule)}
        path = atomic_json(Path(output)/f'seed_{seed}'/'row_refresh.json',row)
        raw[path.relative_to(output).as_posix()] = sha256_file(path)
        fit_rows[seed] = row
    placements = {p.panel_id:p.placement for p in dev if p.panel_id != SITE}
    transfers = paired_site_gains(other,design['seeds'],placements,ORDERS,repetitions=design['bootstrap_draws'],rng_seed=design['bootstrap_seed'])
    return {'source_type':'real_layer','status':'STUDY_COMPLETE','evidence_kind':'MATCHED_ROW_REFRESH','design':design,
            'row_artifacts':raw,'crossfit_reference_sha256':sha256_file(crossfit_archive),'reference_archive_sha256':sha256_file(reference),
            'refresh_summary':summarize_refresh(fit_rows,design),'other_eight_sites':transfers}
