import numpy as np
import pytest

torch = pytest.importorskip('torch')
from cellquant.bilateral_rows import SITE, row_energies
from cellquant.quantization import QuantSpec
from cellquant.row_crossfit import GROUPS, alternate_row_ids, paired_row_effect
from cellquant.solver import CalibrationPanel


def test_swap_is_disjoint_and_closes_each_projected_error_on_all_rows():
    count = 20
    sampled = np.linspace(0,count-1,4,dtype=int).tolist()
    alternate = alternate_row_ids(count,sampled,11)
    assert alternate == alternate_row_ids(count,sampled,11)
    assert all(i//5 == stratum for stratum,i in enumerate(alternate))
    assert set(sampled).isdisjoint(alternate)
    w = torch.arange(count*8,dtype=torch.float32).reshape(count,8).sin()
    x = torch.arange(8*5,dtype=torch.float32).reshape(8,5).cos()
    q = torch.eye(8,dtype=torch.float64)
    panel = CalibrationPanel(SITE,'input',w[sampled],x,metadata={'weight_row_ids':sampled,'tokens_per_block':[2,3],
        'document_ids':['one','two'],'block_ids':['one:0','two:0']})
    swapped = row_energies(w,panel,q,QuantSpec(group_size=4),alternate_rows=alternate,tile_rows=3)
    original = row_energies(w,panel,q,QuantSpec(group_size=4))
    assert swapped['row_counts'] == dict(sampled=4,alternate=4,complement=12,all_rows=20)
    for i, block in enumerate(original['blocks']['all_rows']):
        for key in ('W4A4','W4Afp','WfpA4','target_energy'):
            assert sum(swapped['blocks'][group][i][key] for group in GROUPS[:3]) == pytest.approx(block[key])
    with pytest.raises(ValueError,match='disjoint'):
        row_energies(w,panel,q,QuantSpec(),alternate_rows=sampled)


def test_interaction_uses_same_documents_for_both_row_sets():
    # Each article has its own baseline energy. B-vs-A gains are -20% on A, +10% on B for every resample.
    by_seed = {}
    for seed in (11,23):
        methods = {}
        for method in ('grid_a','grid_b'):
            blocks = {}
            for group, ratio in [('sampled',1.2),('alternate',.9)]:
                blocks[group] = [{'document_id':f'{seed}_{i}', 'block_id':f'{seed}_{i}:0','sampled_tokens':2,
                    'target_energy':10*(i+1), 'W4A4':(i+1)*(ratio if method=='grid_b' else 1.)} for i in range(3)]
            methods[method] = {'blocks':blocks}
        by_seed[seed] = {'methods':methods}
    options = dict(draws=100,rng_seed=7,difference=('alternate','sampled'))
    effect = paired_row_effect(by_seed,'grid_b','grid_a',['sampled','alternate'],'W4A4',**options)
    assert effect['difference_pp']['paired_95'] == pytest.approx([30.,30.])
    by_seed[11]['methods']['grid_a']['blocks']['alternate'][0]['sampled_tokens'] = 1
    with pytest.raises(ValueError,match='Unpaired'):
        paired_row_effect(by_seed,'grid_b','grid_a',['sampled','alternate'],'W4A4',**options)


def test_refresh_averages_matched_orders_in_each_draw_and_requires_feasible_endpoints():
    from cellquant.row_refresh import DESIGN, summarize_refresh
    rows = {}
    for seed in DESIGN['seeds']:
        methods = {}
        # The two denominators differ. Pooled errors or picking the winning order gives a different effect.
        for label,errors in {'aa':[10.,10.],'ab':[7.,9.],'bb':[20.,20.],'ba':[25.,21.]}.items():
            methods[label] = {'blocks':{g:[{'document_id':f'{seed}/{j}','block_id':f'{seed}/{j}:0',
                'sampled_tokens':32,'target_energy':100.,'W4A4':e} for j,e in enumerate(errors)] for g in GROUPS}}
        rows[seed] = {'methods':methods,'tails':{m:{'destination_feasible_at_end':True} for m in ['ab','ba']}}
    result = summarize_refresh(rows,{**DESIGN,'bootstrap_draws':1000})
    effect = result['mean_paired_orders']['complement']
    # Each draw gives either 2.5% or a weighted average of 2.5% with itself.
    assert effect['estimate'] == pytest.approx(2.5)
    assert effect['paired_95'] == pytest.approx([2.5,2.5])
    assert result['next_step'] == 'replicate_refresh_on_more_projections'
    rows[11]['tails']['ba']['destination_feasible_at_end'] = False
    assert summarize_refresh(rows,{**DESIGN,'bootstrap_draws':100})['next_step'] == 'retain_row_effect_stop_single_switch'


def test_cross_layer_gap_pairs_articles_and_excludes_the_discovery_layer():
    from cellquant.bilateral_rows import PRECISIONS, ROW_SETS
    from cellquant.row_transfer import summarize_transfer, transfer_design
    design = {**transfer_design(),'bootstrap_draws':200}
    rows = {}
    for seed in design['seeds']:
        sites = {}
        for layer,ratio in zip(design['layers'],[2.,0.8,0.9]):
            for projection in design['projections']:
                site = f'layer_{layer}.{projection}'
                sites[site] = {}
                for name in ('full','short'):
                    data = {}
                    for group in ROW_SETS:
                        fit = group=='sampled' and name=='full'
                        multiplier = (ratio if projection=='up_proj' else 1.4) if fit else 1.
                        data[group] = [{'document_id':f'{seed}/{j}','block_id':f'{seed}/{j}:0','sampled_tokens':2,
                                        'target_energy':10*j,**{p:j*multiplier for p in PRECISIONS}} for j in (1,2)]
                    sites[site][name] = {'blocks':data}
        rows[seed] = {'sites':sites}
    report = summarize_transfer(rows,design)
    assert report['effects']['W4A4']['mean_middle_last_up_gap_pp']['paired_95'] == pytest.approx([15.,15.])
    assert report['next_step'] == 'confirm_with_independent_row_swap_and_third_seed'
    for b in rows[23]['sites']['layer_31.up_proj']['full']['blocks']['sampled']:
        b['W4A4'] *= 1.2
    assert summarize_transfer(rows,design)['next_step'] == 'limit_claim_to_replicated_sites'
