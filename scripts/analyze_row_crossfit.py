"""Recompute the 7B row interaction and transfer effects from paired document energies."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from cellquant.bilateral_documents import COHORT_FILE as PRIOR_COHORT, FIT_FILE, GRID_FILE, IDENTITY, block_identity, checked_json, read_7b_fit
from cellquant.bilateral_rows import PRECISIONS
from cellquant.data import token_hash
from cellquant.matching_study import paired_site_gains
from cellquant.row_crossfit import COHORT_FILE, DESIGN, GROUPS
from cellquant.row_refresh import read_crossfit_archive, replay_grids
from cellquant.runtime import atomic_json, config_hash, read_json, sha256_file
from cellquant.solver import pair_schedule


def paired_sums(rows, design=DESIGN):
    """Independent implementation for disjoint seed strata, one block per article."""
    # Strata in the registered sampling code are visited in global document-ID order.
    seeds = sorted(rows, key=lambda seed:min(rows[seed]['heldout_documents']))
    rng = np.random.default_rng(design['bootstrap_seed'])
    result = {}
    for seed in seeds:
        docs = sorted(rows[seed]['heldout_documents'])
        n = len(docs)
        weights = rng.multinomial(n,np.full(n,1/n),size=design['bootstrap_draws'])
        weights = np.vstack([np.ones(n),weights])
        methods = {}
        for method, measurement in rows[seed]['methods'].items():
            methods[method] = {}
            for group, blocks in measurement['blocks'].items():
                ordered = sorted(blocks,key=lambda b:b['document_id'])
                if [b['document_id'] for b in ordered] != docs:
                    raise ValueError('A row group does not contain the paired articles')
                values = np.array([[b['target_energy'],*(b[p] for p in PRECISIONS)] for b in ordered])
                if not np.isfinite(values).all() or (values[:,0] <= 0).any() or (values[:,1:] < 0).any():
                    raise ValueError('Invalid document energy')
                totals = weights @ values
                methods[method][group] = totals[:,1:]/totals[:,:1]
        result[seed] = methods
    return result


def check_contrast(sums, new, old, precision, expected, *, interaction=False):
    i = PRECISIONS.index(precision)
    by_group = {g:np.array([100*(1-by[new][g][:,i]/by[old][g][:,i]) for _,by in sorted(sums.items())]) for g in GROUPS}
    if interaction:
        by_group['difference_pp'] = by_group['alternate'] - by_group['sampled']
    for group, values in by_group.items():
        center = values.mean(axis=0)
        actual = [center[0], *np.percentile(center[1:],[2.5,97.5]), *values[:,0]]
        saved = expected[group]
        archived = [saved['estimate'], *saved['paired_95'], *(saved['by_seed'][str(s)] for s in sorted(sums))]
        np.testing.assert_allclose(actual,archived,rtol=1e-10,atol=1e-10)


def audit(path, refs, *, replay_cuda=False):
    grid = refs/GRID_FILE
    fit = read_7b_fit(refs/FIT_FILE,grid)
    result,rows = read_crossfit_archive(path,grid,small=False)
    cross_cohort,prior = read_json(refs/COHORT_FILE),read_json(refs/PRIOR_COHORT)
    ever = [*fit['prior'],*prior['previous_42m_train_blocks'],*(b for p in fit['parts'].values() for bs in p.values() for b in bs),
            *(b for bs in prior['heldout'].values() for b in bs)]
    heldout = [b for bs in cross_cohort['heldout'].values() for b in bs]
    for key in IDENTITY:
        seen = {b[key] for b in ever}
        fresh = {b[key] for b in heldout}
        if seen & fresh or len(fresh) != 32:
            raise ValueError('Held-out material reused across experiments or calibration seeds')
    other, timings, observations = [], {}, {}
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Damaged study archive')
        cfg = json.loads(archive.read('run/config.json'))
        protocol = checked_json(archive,'inputs/protocol.json',result['protocol_sha256'])
        fingerprint = json.loads(archive.read('run/execution_fingerprint.json'))
        cohort_copy = checked_json(archive,'inputs/fit_reference/'+COHORT_FILE,cfg['row_crossfit']['cohort_sha256'])
        parent = fit['config']
        if (config_hash(cfg) != result['config_sha256'] or fingerprint['status'] != 'VERIFIED'
                or config_hash(fingerprint['payload']) != fingerprint['sha256'] or not fingerprint['payload']['code_git']['tracked_clean']
                or cfg['row_crossfit'] != protocol['row_crossfit'] or protocol['row_crossfit']['design'] != DESIGN
                or cohort_copy != cross_cohort or cross_cohort['previous_cohort_hash'] != config_hash(prior)
                or any(cfg[k] != parent[k] for k in ('model_revision','dataset_revision','tokenizer_revision','rotation','quantization','formal_pilot','solver'))
                or any(hashlib.sha256(archive.read('inputs/fit_reference/'+name)).hexdigest() != sha256_file(refs/name) for name in (GRID_FILE,FIT_FILE))):
            raise ValueError('Frozen study, manifest or code fingerprint differs')
        for seed,row in rows.items():
            manifest = checked_json(archive,f'inputs/seed_{seed}/data_manifest.json',protocol['token_manifest_sha256'][str(seed)])
            parts = manifest['partitions']
            actual = [block_identity(b) for b in parts['dev']]
            if (actual != cross_cohort['heldout'][str(seed)] or row['heldout_documents'] != [b['document_id'] for b in actual]
                    or [block_identity(b) for b in parts['fit']] != [block_identity(b) for b in fit['parts'][seed]['fit']]
                    or sorted(row['fit_documents']) != fit['grids'][seed]['fit_documents']
                    or row['fit_a_panel_sha256'] != fit['grids'][seed]['fit_panel_fingerprint']
                    or row['fit_a_panel_sha256'] == row['fit_b_panel_sha256']):
                raise ValueError('Panel or document identity differs')
            tokens = archive.read(f'inputs/seed_{seed}/token_ids.npz')
            if hashlib.sha256(tokens).hexdigest() != manifest['tokens_sha256']:
                raise ValueError('Token archive changed')
            with np.load(io.BytesIO(tokens),allow_pickle=False) as cache:
                if any(token_hash(cache[b['array_key']]) != b['token_ids_sha256'] for bs in parts.values() for b in bs):
                    raise ValueError('Token blocks do not match their digests')
            keys = ['target_energy',*PRECISIONS]
            ids = [(b['document_id'],b['block_id'],DESIGN['tokens_per_document']) for b in actual]
            targets = None
            expected_placements = {p['panel_id']:p['placement'] for p in fit['grids'][seed]['heldout'] if p['panel_id'] != DESIGN['site']}
            for method,measured in row['methods'].items():
                if measured['row_counts'] != dict(sampled=64,alternate=64,complement=10880,all_rows=11008):
                    raise ValueError('Full projection partition changed')
                for blocks in measured['blocks'].values():
                    if [(b['document_id'],b['block_id'],b['sampled_tokens']) for b in blocks] != ids:
                        raise ValueError('Rows, token counts and documents are unpaired')
                values = np.array([[[b[k] for k in keys] for b in measured['blocks'][g]] for g in GROUPS])
                np.testing.assert_allclose(values[:3].sum(axis=0),values[3],rtol=1e-12)
                if targets is None:
                    targets = values[:,:,0]
                np.testing.assert_allclose(values[:,:,0],targets,rtol=1e-12)
                if method.startswith('grid_'):
                    panels = measured['other_panels']
                    placements = {p['panel_id']:p['placement'] for p in panels}
                    if len(placements) != 8 or placements != expected_placements:
                        raise ValueError('Incomplete other-site measurements')
                    other.append({'seed':seed,'method':method,'panels':{p['panel_id']:p['blocks'] for p in panels}})
            records = row['methods']
            observations[str(seed)] = {group:sum(b['W4A4'] < a['W4A4'] for a,b in zip(records['grid_a']['blocks'][group],records['grid_b']['blocks'][group]))
                                       for group in GROUPS}
            timings[str(seed)] = {'grid_a_seconds_prior_run':fit['grids'][seed]['total_calibration_seconds'],
                                  'grid_b_seconds_this_run':row['grid_b_calibration_seconds']}
            if replay_cuda:
                schedule = pair_schedule(4096,256,seed)
                for arm,stages in [('a',fit['grids'][seed]['solver']['grid_stages']),('b',row['grid_b_stages'])]:
                    replay_grids(stages,4096,seed,'cuda',arm,row,schedule)
    sums = paired_sums(rows)
    summary = result['crossfit_summary']
    for p in PRECISIONS:
        check_contrast(sums,'grid_b','grid_a',p,summary['grid_b_vs_grid_a'][p],interaction=True)
    for arm in ('a','b'):
        check_contrast(sums,'grid_'+arm,'prefix_'+arm,'W4A4',summary['full_vs_short_within_fit'][arm])
    primary = summary['grid_b_vs_grid_a']['W4A4']
    observed = primary['difference_pp']['paired_95'][0] > 0 and all(
        primary['sampled']['by_seed'][str(s)] < 0 < primary['alternate']['by_seed'][str(s)] for s in rows)
    if observed != summary['interaction_replicated']:
        raise ValueError('Published decision differs from the paired measurements')
    sites = paired_site_gains(other,sorted(rows),placements,{'grid_b_vs_grid_a':('grid_b','grid_a')},
                              repetitions=DESIGN['bootstrap_draws'],rng_seed=DESIGN['bootstrap_seed'])
    recorded = result['other_eight_sites']
    if any(sites[k] != recorded[k] for k in sites if k != 'comparisons'):
        raise ValueError('Other-site sampling design changed')
    comparison = sites['comparisons']['grid_b_vs_grid_a']
    archived = recorded['comparisons']['grid_b_vs_grid_a']
    for group in ('all','input','output'):
        for key in ('median_gain_pct','paired_95_pct'):
            np.testing.assert_allclose(comparison[group][key],archived[group][key],atol=1e-10,rtol=1e-10)
        for seed in rows:
            np.testing.assert_allclose(comparison[group]['seed_medians_pct'][str(seed)],archived[group]['seed_medians_pct'][str(seed)],atol=1e-10)
    for site in placements:
        np.testing.assert_allclose(comparison['site_gain_pct'][site],archived['site_gain_pct'][site],atol=1e-10)
    return {'run_id':result['run_id'],'source_type':'real_layer','archive_sha256':sha256_file(path),'site':DESIGN['site'],
            'scope':'two fixed calibration seeds, 32 disjoint new articles, fixed A/B row sets at one projection',
            'device':fingerprint['payload']['environment']['cuda']['devices'][0]['name'],'code_git':fingerprint['payload']['code_git'],
            'wall_seconds':result['resources']['wall_seconds'],'by_seed_timing':timings,'prior_distinct_documents':len({b['document_id'] for b in ever}),
            'grid_b_winning_document_counts':observations,'replayed_all_eight_rotation_hashes_on_cuda':replay_cuda,
            'max_normalized_orthogonality_residual':max(m['orthogonality_residual'] for row in rows.values() for m in row['methods'].values()),
            'crossfit_summary':summary,'other_eight_sites':sites}


def plot(summary, small_path, output):
    import matplotlib.pyplot as plt
    with ZipFile(small_path) as small:
        small_results = json.loads(small.read('result.json'))['crossfit_summary']
    fig,axes=plt.subplots(1,2,figsize=(8,3.1),layout='constrained')
    models = [('42M',small_results,'#18758a',0.08),('7B',summary['crossfit_summary'],'#a94d24',-0.08)]
    for ax,groups in zip(axes,[('sampled','alternate'),('complement','all_rows')]):
        ax.axvline(0,color='0.55',linestyle='--',linewidth=1)
        for label,res,color,offset in models:
            for y,group in enumerate(groups):
                e=res['grid_b_vs_grid_a']['W4A4'][group]
                m=e['estimate']
                lo,hi=e['paired_95']
                ax.errorbar(m,1-y+offset,xerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color=color,label=label if y==0 else None)
        ax.set(xlabel='Relative NMSE reduction (%)',yticks=[1,0],ylim=(-.38,1.35))
        ax.spines[['top','right']].set_visible(False)
    axes[0].set(yticklabels=['Rows fitted by A','Rows fitted by B'],title='Fitted rows')
    axes[1].set(yticklabels=['Rows fitted by neither','Full projection'],title='Transfer to the projection')
    fig.suptitle('W4A4 error of grid B relative to grid A, first MLP up projection')
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='outside lower center',frameon=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    refs = Path(__file__).resolve().parents[1]/'evidence/output_matching_20261119'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',nargs='?',type=Path,default=refs/'row_crossfit_7b.zip')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--figure',type=Path)
    parser.add_argument('--replay-cuda',action='store_true',help='Also rebuild the archived FP64 rotations with their original CUDA RNG')
    args = parser.parse_args()
    result = audit(args.archive,refs,replay_cuda=args.replay_cuda)
    print(atomic_json(args.output,result))
    if args.figure:
        plot(result,refs/'row_crossfit_42m.zip',args.figure)
