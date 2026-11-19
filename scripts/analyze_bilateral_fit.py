"""Audit the A100 fit-only measurements and replay each frozen rotation on CUDA."""
import argparse
from pathlib import Path

import numpy as np
import torch

from cellquant.bilateral_documents import DESIGN, FIT_FILE, GRID_FILE, read_7b_fit
from cellquant.bilateral_screen import prefix_rotation, replay_angle_arms, replay_retest_parent
from cellquant.matching_batches import q_digest
from cellquant.output_matching import fits_limits
from cellquant.pilot import pilot_solver_config
from cellquant.runtime import atomic_json, sha256_file


def audit(screen, grid, device):
    fit = read_7b_fit(screen, grid)
    site = DESIGN['watch_site']
    by_seed, site_gains = {}, []
    tolerance = pilot_solver_config(fit['config']).budget_tolerance
    for seed, ref in fit['grids'].items():
        record = fit['traces'][seed]
        grid_q = replay_retest_parent(ref, device)
        prefix, _ = prefix_rotation(len(grid_q), seed, ref['solver']['grid_stages'][0], device)
        rotations = replay_angle_arms(prefix, record)
        old, new = (record['arms'][m]['selection'] for m in fit['receipt']['design']['arms'])
        for selection in (old, new):
            events = selection['trace']
            if (len(events) != selection['batch_count'] or selection['accepted_batches'] != [t['batch'] for t in events if t['accepted']]
                    or any(t['feasible'] != fits_limits(t, selection['limits'], tolerance)
                           or t['accepted'] != (t['feasible'] and t['loss'] < t['parent_loss']) for t in events)
                    or not np.isclose(sum(selection['final_site_nmse'].values()), selection['final']['loss'], rtol=1e-10)):
                raise ValueError('Raw batch losses, risks or final site losses disagree with the selection')
        def gain(a, b):
            return 100 * (1 - a/b)
        n, o, g = new['final'], old['final'], record['full_grid_fit']
        target = record['full_grid_site_nmse'][site]
        sites = record['summary']['gain_vs_full_grid_by_site_pct']
        site_gains.append(sites)
        by_seed[str(seed)] = {
            'median_site_gain_vs_full_grid_pct':float(np.median(list(sites.values()))),
            'sites_within_two_pct_of_full_grid':sum(v >= -2 for v in sites.values()),
            'joint_fit_gain_vs_output_only_pct':gain(n['loss'],o['loss']), 'joint_fit_gain_vs_full_grid_pct':gain(n['loss'],g['loss']),
            'input_sum_gain_vs_output_only_pct':gain(n['loss']-n['output_loss'],o['loss']-o['output_loss']),
            'output_sum_gain_vs_output_only_pct':gain(n['output_loss'],o['output_loss']),
            'accepted_batches':{m:len(a['selection']['accepted_batches']) for m,a in record['arms'].items()},
            'applied_pairs':{m:a['selection']['applied_pairs'] for m,a in record['arms'].items()},
            'incremental_solver_seconds':record['summary']['incremental_solver_seconds'],
            'prefix_gain_vs_full_grid_at_watched_site_pct':gain(new['initial_site_nmse'][site],target),
            'new_gain_vs_full_grid_at_watched_site_pct':gain(new['final_site_nmse'][site],target),
            'new_gain_vs_output_only_at_watched_site_pct':gain(new['final_site_nmse'][site],old['final_site_nmse'][site]),
            'replayed_q_sha256':{'continued_grid':q_digest(grid_q),'prefix':q_digest(prefix),
                                 **{m:q_digest(q) for m,q in rotations.items()}}}
        del grid_q, prefix, rotations
    per_site = {k:float(np.mean([s[k] for s in site_gains])) for k in site_gains[0]}
    return {'source_type':'real_layer','scope':'A100 fit panels only; no fresh 7B evaluation',
            'run_id':fit['receipt']['run_id'],'next_step':fit['receipt']['next_step'],
            'fit_archive_sha256':sha256_file(screen),'grid_archive_sha256':sha256_file(grid),'watched_site':site,
            'local_replay_device':torch.cuda.get_device_name(device),'by_seed':by_seed,
            'site_gains_vs_grid_pct':per_site,'median_site_gain_vs_grid_pct':float(np.median(list(per_site.values())))}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]/'evidence/output_matching_20261119'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',type=Path,default=root/FIT_FILE)
    parser.add_argument('--grid',type=Path,default=root/GRID_FILE)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out = atomic_json(args.output,audit(args.fit,args.grid,'cuda:0'))
    print(out)
