"""Measure fitted-row and whole-projection gains along one frozen shared-grid trajectory."""
from pathlib import Path

import numpy as np
import torch

from .bilateral_rows import PRECISIONS, ROW_SETS, SITE, row_energies
from .bilateral_screen import prefix_rotation, replay_retest_parent
from .matching_batches import load_matching_arms, q_digest, replay_continued_grid
from .matching_study import WARM
from .progress import report
from .row_crossfit import paired_row_effect
from .row_refresh import read_crossfit_archive, read_refresh_archive
from .runtime import atomic_json, sha256_file
from .solver import panel_fingerprint


REFERENCE_FILE = 'row_refresh_7b.zip'


def transfer_design(small=False):
    return {'version':1,'layers':[0,3,7] if small else [0,15,31],'projections':['up_proj','q_proj'],
            'seeds':[11,23],'rows_per_fit':64,'short_pairs':8 if small else 64,'full_pairs':32 if small else 256,
            'bootstrap_draws':10000,'bootstrap_seed':2026111908,'documents':'same 32 articles as the row-refresh study',
            'primary':'W4A4 full-vs-short gain on fitted rows minus gain on all remaining rows, equally weighted across middle and last up projections and fixed seeds',
            'advance':'paired primary lower 95% bound positive and fitted-minus-remaining gain positive in each of the four layer/seed cells'}


def full_weights(model, design):
    weights = {}
    for layer_id in design['layers']:
        layer = model.model.layers[layer_id]
        for name in design['projections']:
            module = layer.mlp.up_proj if name == 'up_proj' else layer.self_attn.q_proj
            weights[f'layer_{layer_id}.{name}'] = module.module.weight.detach()
    return weights


def summarize_transfer(rows, design):
    sites = [f'layer_{n}.{p}' for n in design['layers'] for p in design['projections']]
    confirm = [f'layer_{n}.up_proj' for n in design['layers'][1:]]
    groups = [f'{site}/{region}' for site in sites for region in ROW_SETS]
    gaps = {site:[(site+'/sampled',site+'/complement')] for site in sites}
    gaps['primary'] = [gaps[site][0] for site in confirm]
    paired = {}
    for seed,record in rows.items():
        methods = {}
        for name in ('short','full'):
            blocks = {f'{site}/{region}':values for site in sites for region,values in record['sites'][site][name]['blocks'].items()}
            methods[name] = {'blocks':blocks}
        paired[seed] = {'methods':methods}
    effects = {}
    for precision in PRECISIONS:
        values = paired_row_effect(paired,'full','short',groups,precision,draws=design['bootstrap_draws'],
                                   rng_seed=design['bootstrap_seed'],gap_groups=gaps)
        effects[precision] = {'mean_middle_last_up_gap_pp':values['primary'], 'sites':{
            site:{**{g:values[site+'/'+g] for g in ROW_SETS},'fitted_minus_remaining_pp':values[site]} for site in sites}}
    primary = effects['W4A4']['mean_middle_last_up_gap_pp']
    confirm_gaps = [effects['W4A4']['sites'][site]['fitted_minus_remaining_pp'] for site in confirm]
    if any('undefined' in x for x in [primary,*confirm_gaps]):
        decision = 'insufficient_evidence'
    elif primary['paired_95'][0] > 0 and all(min(x['by_seed'].values()) > 0 for x in confirm_gaps):
        decision = 'confirm_with_independent_row_swap_and_third_seed'
    else:
        decision = 'limit_claim_to_replicated_sites'
    return {'effects':effects,'next_step':decision}


def check_reference(site, scored, fixed):
    if site == SITE:
        # The earlier experiment also split off B. Its full sum still covers exactly these rows.
        for region in ('sampled','all_rows'):
            assert_blocks(scored['blocks'][region],fixed['blocks'][region],'W4A4')
    else:
        record = next(p for p in fixed['other_panels'] if p['panel_id'] == site)
        assert_blocks(scored['blocks']['sampled'],record['blocks'],'squared_error')


def assert_blocks(actual, reference, error_key):
    keys = ('document_id','block_id','sampled_tokens')
    if (len(reference) != len(actual) or any(any(a[k] != b[k] for k in keys) for a,b in zip(actual,reference))
            or not np.allclose([[a['W4A4'],a['target_energy']] for a in actual],
                               [[b[error_key],b['target_energy']] for b in reference],rtol=1e-9,atol=1e-13)):
        raise ValueError('Fixed grid does not reproduce the archived per-document energies')


@torch.inference_mode()
def run_transfer(config, prepared, weights, output, grid, crossfit, refresh, *, fit_7b=None, progress=None):
    small = fit_7b is None
    design = transfer_design(small)
    if config['row_transfer'] != {'design':design,'refresh_sha256':sha256_file(refresh)} or config['seeds'] != design['seeds']:
        raise ValueError('Cross-layer design or reference changed')
    receipt, references = read_refresh_archive(refresh,crossfit,grid,small=small)
    _, originals = read_crossfit_archive(crossfit,grid,small=small)
    arms = load_matching_arms(grid,design['seeds']) if small else None
    hidden, device = prepared['hidden_size'], config['solver']['device']
    rows, digests = {}, {}
    for seed in design['seeds']:
        ready = prepared['seeds'][seed]
        dev = ready['heldout']
        reference = references[seed]
        original = originals[seed]['methods']
        full_record = reference['methods']['aa']
        if small:
            warm = arms[seed][WARM]['solver']
            full = replay_continued_grid(arms[seed],hidden,seed,device)
        else:
            fitted = fit_7b['grids'][seed]
            if any(p.metadata['graph_identity'] != fitted['graph_identity'] for p in dev):
                raise ValueError('Captured graph differs from the frozen grid')
            warm = fitted['solver']['grid_stages'][0]
            full = replay_retest_parent(fitted,device)
        short, length = prefix_rotation(hidden,seed,warm,device)
        hashes = {'short':q_digest(short),'full':q_digest(full)}
        panels = {p.panel_id:p for p in dev if p.placement=='input'}
        expected = {f'layer_{n}.{p}' for n in design['layers'] for p in design['projections']}
        if (set(weights) != expected or set(panels) != expected or len(dev) != 9 or length != design['short_pairs']
                or hashes != {'short':original['prefix_a']['q_sha256'],'full':original['grid_a']['q_sha256']}
                or hashes['full'] != full_record['q_sha256'] or panel_fingerprint(dev) != reference['heldout_panel_sha256']):
            raise ValueError('Grid endpoints, projections or evaluation panels differ from the archived runs')
        sites = {}
        for site,panel in panels.items():
            ids = panel.metadata['weight_row_ids']
            if (panel.metadata['source_type'] != 'real_layer' or panel.metadata['document_ids'] != reference['heldout_documents']
                    or ids != np.linspace(0,len(weights[site])-1,design['rows_per_fit'],dtype=int).tolist()):
                raise ValueError('Unexpected fitted rows or held-out articles')
            sites[site] = {'fitted_row_ids':ids}
            for name,q in [('short',short),('full',full)]:
                report(progress,'row_transfer.score',seed=seed,site=site,checkpoint=name)
                sites[site][name] = row_energies(weights[site],panel,q,ready['specs'])
            check_reference(site,sites[site]['full'],full_record)
        row = {'seed':seed,'source_type':'real_layer','heldout_documents':reference['heldout_documents'],'sites':sites,
               'rotation_sha256':hashes,'heldout_panel_sha256':reference['heldout_panel_sha256']}
        path = atomic_json(Path(output)/f'seed_{seed}'/'row_transfer.json',row)
        digests[path.relative_to(output).as_posix()] = sha256_file(path)
        rows[seed] = row
    return {'source_type':'real_layer','status':'STUDY_COMPLETE','evidence_kind':'FROZEN_CROSS_LAYER_ROW_TRANSFER',
            'design':design,'row_artifacts':digests,'reference_archive_sha256':sha256_file(refresh),
            'parent_decision':receipt['refresh_summary']['next_step'],'row_transfer_summary':summarize_transfer(rows,design)}
