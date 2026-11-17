"""Reproducible CUDA throughput diagnostic; synthetic data never qualifies G1."""
from pathlib import Path
import argparse
import importlib.util
import json
import time

import torch

from cellquant import tensor_solver
from cellquant.quantization import QuantSpec
from cellquant.runtime import atomic_json,environment_report,sha256_file
from cellquant.solver import CalibrationPanel,SolverConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--case',choices=['small','full_output','mixed_full'],default='full_output')
    parser.add_argument('--method',choices=['cellquant','fixed_scale_search_matched'],default='cellquant')
    parser.add_argument('--seconds-per-pair',type=float,default=30.)
    parser.add_argument('--pairs',type=int,default=1)
    parser.add_argument('--reference-module',help='Optional archived tensor_solver.py for same-GPU comparison')
    args=parser.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU substitution')
    sources=['src/cellquant/tensor_solver.py','src/cellquant/tensor_cells.py','src/cellquant/tensor_math.py','src/cellquant/solver.py',__file__]
    source_hashes={p:sha256_file(p) for p in sources}
    engine=tensor_solver
    if args.reference_module:
        spec=importlib.util.spec_from_file_location('cellquant._archived_tensor_solver',args.reference_module)
        engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    generator=torch.Generator(device='cuda').manual_seed(861)
    def rand(shape):return torch.randn(shape,dtype=torch.float64,device='cuda',generator=generator)
    d,ffn,tokens=(8,16,4) if args.case=='small' else (4096,11008,128)
    panels=[]
    if args.case=='small':panels.append(CalibrationPanel('synthetic.input','input',rand((2,d)),rand((d,tokens))))
    panels.append(CalibrationPanel('synthetic.output','output',rand((d,ffn)),rand((ffn,tokens))))
    if args.case=='mixed_full':
        for index in range(6):panels.append(CalibrationPanel(f'synthetic.input.{index}','input',rand((64,d)),rand((d,tokens))))
    cfg=SolverConfig(backend='torch',device='cuda',seconds_per_pair=args.seconds_per_pair,
                     wall_seconds=max(90.,args.seconds_per_pair*args.pairs+60.),
                     theta_limit=.08 if args.case=='small' else .7853981633974483)
    schedule=[(i%d,(i+1)%d) for i in range(args.pairs)]
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def progress(phase,**details):print(json.dumps({'phase':phase,**details}),flush=True)
    result=engine.fit_baseline(args.method,panels,torch.eye(d,dtype=torch.float64,device='cuda'),QuantSpec(group_size=-1),schedule,cfg,progress=progress)
    result.pop('q')
    report={'source_type':'synthetic_math','evidence_kind':'THROUGHPUT_DIAGNOSTIC_ONLY','case':args.case,
            'scope':'Full matrix axes and angle domain where requested; random data, identity Q, not pretrained layers or G1. mixed_full has six input panels and ONE output panel.',
            'environment':environment_report(),'elapsed_seconds':time.perf_counter()-started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'solver':result,
            'engine_source':str(args.reference_module or 'src/cellquant/tensor_solver.py'),
            'source_hashes':source_hashes,
            'sources_unchanged_during_run':all(sha256_file(p)==h for p,h in source_hashes.items())}
    if args.reference_module:report['reference_module_sha256']=sha256_file(args.reference_module)
    atomic_json(out,report)
    print(json.dumps({k:result.get(k) for k in ['status','reason','seconds','completed_pairs','resource_diagnostics']},indent=2),flush=True)


if __name__=='__main__':main()
