from pathlib import Path
import argparse,json,time
import torch
from cellquant.solver import CalibrationPanel,SolverConfig
from cellquant.quantization import QuantSpec
from cellquant import tensor_solver as ts
from cellquant.runtime import atomic_json,environment_report,sha256_file
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output)
if out.exists():raise FileExistsError(out)
if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU substitution')
rows=[]
for d,ffn,tokens in [(8,16,4),(32,128,8)]:
    rng=torch.Generator(device='cuda').manual_seed(861)
    def rand(shape):return torch.randn(shape,device='cuda',dtype=torch.float64,generator=rng)
    panels=[CalibrationPanel('synthetic.q','input',rand((2,d)),rand((d,tokens)),metadata={'source_type':'synthetic_math'}),
            CalibrationPanel('synthetic.down','output',rand((d,ffn)),rand((ffn,tokens)),metadata={'source_type':'synthetic_math'})]
    cfg=SolverConfig(backend='torch',device='cuda',seconds_per_pair=20,wall_seconds=25,theta_limit=.08)
    for method in ['cellquant','fixed_scale_search_matched']:
        r=ts.fit_baseline(method,panels,torch.eye(d,device='cuda',dtype=torch.float64),QuantSpec(group_size=-1),[(0,1)],cfg)
        r.pop('q');row={'hidden':d,'ffn':ffn,'tokens':tokens,'method':method,**r};rows.append(row)
        print(json.dumps({k:row.get(k) for k in ['hidden','ffn','method','status','reason','seconds','completed_pairs','resource_diagnostics']}),flush=True)
    del panels
    torch.cuda.empty_cache()
atomic_json(out,{'source_type':'synthetic_math','scope':'Same-GPU synthetic performance/timeout probe only; no real G1 evidence',
    'environment':environment_report(),'source_hashes':{s:sha256_file(s) for s in ['src/cellquant/tensor_math.py','src/cellquant/tensor_solver.py']},'runs':rows})
