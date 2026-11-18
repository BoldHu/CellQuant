"""Exercise the matching stage with full LLaMA-2-7B axes and eight synthetic tokens."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from cellquant.output_matching import disjoint_pairs, fit_matching, fit_totals
from cellquant.quantization import QuantSpec
from cellquant.runtime import atomic_json, sha256_file
from cellquant.solver import CalibrationPanel, SolverConfig
from cellquant.tensor_solver import hadamard_rotation


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = 'cuda:0'
    generator = torch.Generator(device=device).manual_seed(11)
    def sample(shape):
        return torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    panels = [
        CalibrationPanel('q', 'input', sample((64, 4096)), sample((4096, 8))),
        CalibrationPanel('down', 'output', sample((4096, 11008)), sample((11008, 8))),
    ]
    torch.cuda.reset_peak_memory_stats()
    q = hadamard_rotation(4096, 11, device)
    pairs = disjoint_pairs(4096, 11)
    specs = QuantSpec(group_size=-1)
    initial = fit_totals(panels, q, specs)
    limits = {k:initial[k] for k in ('budget_a', 'budget_w')}
    refined, result = fit_matching(panels, q, specs, pairs, SolverConfig(device=device),
                                   limits, criterion='product')
    replay = fit_totals(panels, refined, specs)
    upper = result['risk_allocation']['full_step_risk_upper']
    assert replay['loss'] <= initial['loss']
    assert all(replay[k] <= limits[k]+1e-10 for k in limits)
    assert all(result['joint_fit_trials'][-1][k] <= upper[k]+1e-10 for k in upper)
    orth = float(torch.linalg.norm(refined.T @ refined - torch.eye(4096, device=device, dtype=q.dtype))/64)
    assert orth < 1e-12
    payload = {'source_type':'synthetic_math','scope':'Full-size axes, one input and one output panel, eight synthetic tokens',
               'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'weights':[[64,4096],[4096,11008]],
               'fit_tokens':8,'group_size':-1,'matching_pairs':len(pairs),'angles':len(result['grid_radians']),
               'fit':{k:result[k] for k in ('initial','final','limits','risk_allocation','seconds','selected_scale',
                                           'nonzero_applied_angles','product_replay_minus_predicted_change')},
               'orthogonality_residual':orth,'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
               'implementation_sha256':sha256_file(ROOT/'src/cellquant/output_matching.py')}
    atomic_json(args.output, payload)
    print(args.output, payload['fit']['seconds'], payload['peak_cuda_allocated_gib'])


if __name__=='__main__':
    main()
