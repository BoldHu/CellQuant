import numpy as np
import pytest

torch = pytest.importorskip('torch')
from cellquant.bilateral_rows import PREFIX, ROW_SETS, SITE, row_energies, summarize_rows
from cellquant.matching_study import CONTINUED
from cellquant.quantization import QuantSpec, quantize
from cellquant.solver import CalibrationPanel


def test_complement_rows_and_one_sided_products_match_dense_quantization():
    rng = np.random.default_rng(8)
    w = rng.normal(size=(11,8)).astype(np.float32)
    x = rng.normal(size=(8,7)).astype(np.float32)
    q = np.linalg.qr(rng.normal(size=(8,8)))[0]
    rows = [0,4,10]
    panel = CalibrationPanel(SITE, 'input', w[rows], x, metadata={'weight_row_ids':rows,'tokens_per_block':[2,5],
                             'block_ids':['a:0','b:0'],'document_ids':['a','b']})
    wspec, aspec = QuantSpec(group_size=4), QuantSpec(group_size=2)
    result = row_energies(torch.from_numpy(w), panel, torch.from_numpy(q), (wspec,aspec), tile_rows=2)
    wr, xr, target = w.astype(float) @ q.T, q @ x, w.astype(float) @ x
    wh, xh = quantize(wr,wspec)[0], quantize(xr.T,aspec)[0].T
    arrays = {'target_energy':target, 'W4A4':wh @ xh-target, 'W4Afp':wh @ xr-target, 'WfpA4':wr @ xh-target}
    for group, indices in {'sampled':rows, 'complement':np.setdiff1d(np.arange(len(w)),rows), 'all_rows':np.arange(len(w))}.items():
        assert result['row_counts'][group] == len(indices)
        for b, token_slice in zip(result['blocks'][group], (slice(2),slice(2,None))):
            for key, value in arrays.items():
                assert b[key] == pytest.approx((value[indices,token_slice]**2).sum(), rel=1e-10, abs=1e-13)
    changed = torch.from_numpy(w.copy())
    changed[4] *= 2
    with pytest.raises(ValueError, match='captured layer weights'):
        row_energies(changed, panel, torch.from_numpy(q), (wspec,aspec))


def test_exact_single_sided_grid_only_makes_that_ratio_undefined():
    methods = {}
    for method, error in ((CONTINUED,1.), (PREFIX,1.1), ('candidate',.8)):
        blocks = [{'block_id':doc+':0','document_id':doc,'sampled_tokens':2,'target_energy':4.,
                   'W4A4':error, 'W4Afp':error, 'WfpA4':0. if method == CONTINUED else error} for doc in ('x','y')]
        methods[method] = {'blocks':{name:blocks for name in ROW_SETS}}
    effects = summarize_rows({11:{'methods':methods}}, 'candidate', {'bootstrap_draws':30,'bootstrap_seed':11})
    complete = effects['candidate_vs_full_grid']['all_rows']
    assert 'undefined' in complete['WfpA4']
    assert complete['W4A4']['median_gain_pct'] == pytest.approx(20.)
    assert effects['candidate_vs_prefix']['all_rows']['WfpA4']['paired_95_pct'][0] > 0
