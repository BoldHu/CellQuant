"""Verify the midpoint row-switch experiment and recompute its paired document intervals."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from analyze_row_crossfit import paired_sums
from cellquant import bilateral_documents as documents, row_crossfit, row_refresh
from cellquant.bilateral_rows import PRECISIONS
from cellquant.data import token_hash
from cellquant.matching_batches import q_digest
from cellquant.matching_study import paired_site_gains
from cellquant.output_matching import fits_limits
from cellquant.pilot import pilot_solver_config
from cellquant.runtime import atomic_json, config_hash, read_json, sha256_file
from cellquant.solver import pair_schedule
from cellquant.tensor_solver import rotate_rows


def interval(seed_draws, seeds):
    pooled = seed_draws.mean(axis=0)
    return {'estimate':float(pooled[0]), 'paired_95':np.percentile(pooled[1:],[2.5,97.5]).tolist(),
            'by_seed':{str(s):float(v[0]) for s,v in zip(seeds,seed_draws)}}


def compare(actual, archived):
    if isinstance(actual, dict):
        if actual.keys() != archived.keys():
            raise ValueError('Archived fields differ from the recomputed summary')
        for key, value in actual.items():
            compare(value, archived[key])
    elif isinstance(actual, (int,float)) or (isinstance(actual,list) and all(isinstance(v,(int,float)) for v in actual)):
        np.testing.assert_allclose(actual,archived,rtol=1e-10,atol=1e-10)
    elif actual != archived:
        raise ValueError('Archived metadata or decision differs from the measurements')


def error_components(sums, values, seeds):
    """All components use each fixed control's bilateral error as the common denominator."""
    result = {k:{} for k in [*row_refresh.ORDERS,'mean_paired_orders']}
    for group in row_crossfit.GROUPS:
        components = {}
        for label,(new,base) in row_refresh.ORDERS.items():
            delta = np.stack([100*(sums[s][base][group]-sums[s][new][group])/sums[s][base][group][:,:1] for s in seeds])
            bilateral,weight,activation = (delta[:,:,PRECISIONS.index(p)] for p in ('W4A4','W4Afp','WfpA4'))
            np.testing.assert_allclose(bilateral,values[label][group],rtol=1e-10,atol=1e-10)
            components[label] = np.stack([weight,activation,bilateral-weight-activation],axis=-1)
        components['mean_paired_orders'] = np.mean(list(components.values()),axis=0)
        for label,arrays in components.items():
            result[label][group] = {name:interval(arrays[:,:,i],seeds) for i,name in enumerate(
                ('weight_only','activation_only','remainder_including_bilinear_and_cross_terms'))}
    return result


def audit(path, refs, replay_cuda):
    design = row_refresh.DESIGN
    seeds = design['seeds']
    grid, cross = refs/documents.GRID_FILE, refs/row_refresh.REFERENCE_FILE
    fit = documents.read_7b_fit(refs/documents.FIT_FILE,grid)
    _,parents = row_refresh.read_crossfit_archive(cross,grid,small=False)
    result,rows = row_refresh.read_refresh_archive(path,cross,grid,small=False)
    prior = read_json(refs/documents.COHORT_FILE)
    swapped = read_json(refs/row_crossfit.COHORT_FILE)
    cohort = read_json(refs/row_refresh.COHORT_FILE)
    old = [*fit['prior'],*prior['previous_42m_train_blocks'],*cohort['previous_small_heldout'],
           *(b for parts in fit['parts'].values() for blocks in parts.values() for b in blocks),
           *(b for c in (prior,swapped) for blocks in c['heldout'].values() for b in blocks)]
    fresh = [b for blocks in cohort['heldout'].values() for b in blocks]
    for key in documents.IDENTITY:
        seen = {b[key] for b in old}
        new = {b[key] for b in fresh}
        if len(new) != 32 or new & seen:
            raise ValueError(f'Document cohort reused {key}')
    feasible, timings, wins, other = [], {}, {}, []
    pairs = {**row_refresh.ORDERS,'bb_vs_aa':('bb','aa')}
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Incomplete ZIP')
        cfg = json.loads(archive.read('run/config.json'))
        proto = documents.checked_json(archive,'inputs/protocol.json',result['protocol_sha256'])
        fingerprint = json.loads(archive.read('run/execution_fingerprint.json'))
        embedded = archive.read('inputs/fit_reference/'+row_refresh.COHORT_FILE)
        binding = {'design':design,'crossfit_sha256':sha256_file(cross),'cohort_sha256':hashlib.sha256(embedded).hexdigest()}
        if (fingerprint['status'] != 'VERIFIED' or config_hash(fingerprint['payload']) != fingerprint['sha256']
                or config_hash(cfg) != result['config_sha256'] or not fingerprint['payload']['code_git']['tracked_clean']
                or binding != cfg['row_refresh'] or binding != proto['row_refresh']
                or cohort['design'] != design or cohort['previous_cohort_hash'] != config_hash(swapped)
                or any(cfg[k] != fit['config'][k] for k in ('formal_pilot','model_revision','dataset_revision','tokenizer_revision','solver','rotation','quantization','runtime'))):
            raise ValueError('Execution, protocol or parent grid configuration differs')
        for name in (documents.GRID_FILE,documents.FIT_FILE,row_refresh.REFERENCE_FILE):
            if hashlib.sha256(archive.read('inputs/fit_reference/'+name)).hexdigest() != sha256_file(refs/name):
                raise ValueError(f'Reference mismatch for {name}')
        for name,digest in [(documents.COHORT_FILE,cfg['bilateral_document_pilot']['archives'][documents.COHORT_FILE]),
                            (row_crossfit.COHORT_FILE,cfg['row_crossfit']['cohort_sha256']),(row_refresh.COHORT_FILE,binding['cohort_sha256'])]:
            if documents.checked_json(archive,'inputs/fit_reference/'+name,digest) != read_json(refs/name):
                raise ValueError(f'Cohort differs from the registered document list in {name}')
        solver = pilot_solver_config(cfg)
        for seed in seeds:
            row, parent = rows[seed], parents[seed]
            manifest = documents.checked_json(archive,f'inputs/seed_{seed}/data_manifest.json',proto['token_manifest_sha256'][str(seed)])
            raw_tokens = archive.read(f'inputs/seed_{seed}/token_ids.npz')
            parts = manifest['partitions']
            blocks = [documents.block_identity(b) for b in parts['dev']]
            if (hashlib.sha256(raw_tokens).hexdigest() != manifest['tokens_sha256'] or blocks != cohort['heldout'][str(seed)]
                    or [documents.block_identity(b) for b in parts['fit']] != [documents.block_identity(b) for b in fit['parts'][seed]['fit']]
                    or row['heldout_documents'] != [b['document_id'] for b in blocks] or row['fit_documents'] != parent['fit_documents']
                    or any(row['rows_'+a] != parent['rows_'+a] or row['fit_panel_hashes'][a] != parent['fit_'+a+'_panel_sha256'] for a in 'ab')):
                raise ValueError('Unexpected calibration rows, documents or token file')
            with np.load(io.BytesIO(raw_tokens),allow_pickle=False) as arrays:
                if any(token_hash(arrays[b['array_key']]) != b['token_ids_sha256'] for bs in parts.values() for b in bs):
                    raise ValueError('Token array checksum differs')
            schedule = pair_schedule(4096,design['grid_pairs'],seed)
            if config_hash(schedule) != row['pair_schedule_sha256'] or row['pair_schedule_sha256'] != parent['pair_schedule_sha256']:
                raise ValueError('Grid coordinates differ')
            timings[seed], wins[seed] = {}, {}
            targets = None
            for method,rec in row['methods'].items():
                if rec['row_counts'] != dict(sampled=64,alternate=64,complement=10880,all_rows=11008):
                    raise ValueError('Row group size differs')
                for data in rec['blocks'].values():
                    if [(b['document_id'],b['block_id'],b['sampled_tokens']) for b in data] != [(b['document_id'],b['block_id'],32) for b in blocks]:
                        raise ValueError('Unpaired documents')
                v = np.array([[[b[k] for k in ('target_energy',*PRECISIONS)] for b in rec['blocks'][g]] for g in row_crossfit.GROUPS])
                np.testing.assert_allclose(v[:3].sum(axis=0),v[3],rtol=1e-12)
                if targets is None:
                    targets = v[:,:,0]
                np.testing.assert_allclose(targets,v[:,:,0],rtol=1e-12)
                panels = {p['panel_id']:p['blocks'] for p in rec['other_panels']}
                placements = {p['panel_id']:p['placement'] for p in fit['grids'][seed]['heldout'] if p['panel_id'] != row['site']}
                if {p['panel_id']:p['placement'] for p in rec['other_panels']} != placements:
                    raise ValueError('Other projections differ from the fit')
                other.append({'seed':seed,'method':method,'panels':panels})
            half = len(schedule)//2
            for arm,destination in [('a','b'),('b','a')]:
                crossed = arm+destination
                t = row['tails'][crossed]
                tail = t['solver']
                limits = parent['grid_'+destination+'_limits']
                if (row['methods'][arm+arm]['q_sha256'] != parent['methods']['grid_'+arm]['q_sha256']
                        or tail['status'] != 'COMPLETE' or tail['completed_pairs'] != half or tail['limits'] != limits
                        or [v['pair'] for v in tail['log']] != [list(p) for p in schedule[half:]]
                        or any(v['evaluations'] != design['grid_points'] for v in tail['log'])
                        or t['accounted_calibration_seconds'] > solver.wall_seconds):
                    raise ValueError('Rotation source or matched solver budget differs')
                for label,key in [('start','initial'),('end','final')]:
                    observed = fits_limits(tail[key], limits, solver.budget_tolerance)
                    if observed != t['destination_feasible_at_'+label]:
                        raise ValueError('Budget feasibility differs')
                    if label == 'end':
                        feasible.append(observed)
                if replay_cuda:
                    stages = fit['grids'][seed]['solver']['grid_stages'] if arm=='a' else parent['grid_b_stages']
                    q,_ = row_refresh.replay_grids(stages,4096,seed,'cuda',arm,parent,schedule)
                    if q_digest(q) != t['initial_q_sha256']:
                        raise ValueError('Midpoint rotation differs')
                    for step in tail['log']:
                        if step['accepted']:
                            rotate_rows(q, step['pair'], step['best']['theta'])
                    if q_digest(q) != row['methods'][crossed]['q_sha256']:
                        raise ValueError('Crossed rotation differs')
                timings[seed][crossed] = {'accounted_seconds':t['accounted_calibration_seconds'],'accepted_tail_pairs':len(tail['factors'])}
            for label,(new,old_method) in pairs.items():
                wins[seed][label] = {g:sum(b['W4A4'] < a['W4A4'] for a,b in zip(
                    row['methods'][old_method]['blocks'][g],row['methods'][new]['blocks'][g])) for g in row_crossfit.GROUPS}
    sums = paired_sums(rows,design)
    values = {label:{g:np.stack([100*(1-sums[s][new][g][:,0]/sums[s][base][g][:,0]) for s in seeds])
                     for g in row_crossfit.GROUPS} for label,(new,base) in pairs.items()}
    means = {g:interval(np.mean([values[k][g] for k in row_refresh.ORDERS],axis=0),seeds) for g in row_crossfit.GROUPS}
    fixed = {g:interval(v,seeds) for g,v in values['bb_vs_aa'].items()}
    fixed['difference_pp'] = interval(values['bb_vs_aa']['alternate']-values['bb_vs_aa']['sampled'],seeds)
    primary = means['complement']
    advance = all(feasible) and primary['paired_95'][0]>0 and min(primary['by_seed'].values())>0 and means['all_rows']['estimate']>0
    summary = {'by_order':{k:{g:interval(v,seeds) for g,v in values[k].items()} for k in row_refresh.ORDERS},
               'mean_paired_orders':means,'both_orders_feasible':all(feasible),
               'next_step':'replicate_refresh_on_more_projections' if advance else 'retain_row_effect_stop_single_switch'}
    compare(summary,result['refresh_summary'])
    remaining = paired_site_gains(other,seeds,placements,row_refresh.ORDERS,repetitions=design['bootstrap_draws'],rng_seed=design['bootstrap_seed'])
    compare(remaining,result['other_eight_sites'])
    return {'run_id':result['run_id'],'archive_sha256':sha256_file(path),'source_type':'real_layer','scope':'conditional on the two fitted seeds and first up projection',
            'prior_distinct_documents':len({b['document_id'] for b in old}),'distinct_documents':len(fresh),'wall_seconds':result['resources']['wall_seconds'],
            'code_git':fingerprint['payload']['code_git'],'device':fingerprint['payload']['environment']['cuda']['devices'][0]['name'],
            'replayed_rotations_on_cuda':replay_cuda,'calibration':timings,'refresh_summary':summary,'fixed_grid_replication':fixed,
            'wins_by_document':wins,'other_eight_sites':remaining,
            'post_hoc_components_pct_of_control_bilateral_error':error_components(sums,values,seeds)}


def plot(summary, output):
    import matplotlib.pyplot as plt
    estimates = summary['refresh_summary']
    series = [('A → B vs A → A',estimates['by_order']['ab_vs_aa'],'#286482','o'),
              ('B → A vs B → B',estimates['by_order']['ba_vs_bb'],'#ab4525','s')]
    fig, axes = plt.subplots(1,2,figsize=(9,3.25),layout='constrained')
    for ax,groups,labels in zip(axes,[('sampled','alternate'),('complement','all_rows')],
                                [('A rows (64)','B rows (64)'),('Unseen (10,880)','Full (11,008)')]):
        ax.axvline(0,color='0.5',ls='--',lw=1)
        shown = series if ax is axes[0] else series+[('Mean of orders',estimates['mean_paired_orders'],'#202428','D')]
        for j,(label,data,color,marker) in enumerate(shown):
            for i,group in enumerate(groups):
                effect = data[group]
                m = effect['estimate']
                lo,hi = effect['paired_95']
                ax.errorbar(m,1-i+.20*(1-j),xerr=[[m-lo],[hi-m]],fmt=marker,color=color,capsize=3,label=label if i==0 else None)
        ax.set(yticks=[1,0],yticklabels=labels,ylim=(-.4,1.4),xlabel='Relative W4A4 NMSE reduction (%)')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_title('Rows seen during calibration')
    axes[1].set_title('Transfer to the projection')
    fig.suptitle('Midpoint row switch in the first MLP up projection of LLaMA-2-7B')
    fig.legend(*axes[1].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    refs = Path(__file__).resolve().parents[1]/'evidence/output_matching_20261119'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path,nargs='?',default=refs/'row_refresh_7b.zip')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figure',type=Path)
    parser.add_argument('--replay-cuda',action='store_true')
    args = parser.parse_args()
    report = audit(args.archive,refs,args.replay_cuda)
    print(atomic_json(args.output,report))
    if args.figure:
        plot(report,args.figure)
