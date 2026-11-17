from pathlib import Path
import time,json,torch
from cellquant.solver import CalibrationPanel,SolverConfig
from cellquant.quantization import QuantSpec
from cellquant import tensor_solver as ts
from cellquant.runtime import atomic_json,environment_report,sha256_file
out=Path(__file__).with_name('full_output_probe.json')
if out.exists():raise FileExistsError(out)
if not torch.cuda.is_available():raise RuntimeError('CUDA required')
rng=torch.Generator(device='cuda').manual_seed(861)
d,ffn,tokens=4096,11008,128
panel=CalibrationPanel('synthetic.full_down','output',
    torch.randn((d,ffn),device='cuda',dtype=torch.float64,generator=rng),
    torch.randn((ffn,tokens),device='cuda',dtype=torch.float64,generator=rng),metadata={'source_type':'synthetic_math'})
cfg=SolverConfig(backend='torch',device='cuda',seconds_per_pair=30,wall_seconds=90,theta_limit=.7853981633974483)
started=time.perf_counter()
def progress(phase,**details):print(json.dumps({'phase':phase,**details}),flush=True)
r=ts.fit_baseline('cellquant',[panel],torch.eye(d,device='cuda',dtype=torch.float64),QuantSpec(group_size=-1),[(0,1)],cfg,progress=progress)
r.pop('q')
report={'source_type':'synthetic_math','scope':'One full 4096x11008 output panel with 128 tokens; 30 second diagnostic cap, not a real 9-site G1',
    'environment':environment_report(),'elapsed_seconds':time.perf_counter()-started,'solver':r,
    'source_hashes':{s:sha256_file(s) for s in ['src/cellquant/tensor_math.py','src/cellquant/tensor_solver.py']}}
atomic_json(out,report)
print(json.dumps({k:r[k] for k in ['status','reason','seconds','completed_pairs','resource_diagnostics']}),flush=True)
