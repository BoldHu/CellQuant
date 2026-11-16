"""Numerical rotation/gradient comparison; real-model quality remains untested."""
import argparse
import importlib.util
from pathlib import Path
import torch
from cellquant.runtime import atomic_json,environment_report


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.QuantizeLinear


def run(args):
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(11)
    reference=load(Path(args.source)/'train_utils/quant_linear.py','reference_linear')
    port=load(Path(args.overlay)/'train_utils/quant_linear.py','port_linear')
    rows=[]
    for transpose in [False,True]:
        a=reference(16,16,bias=False).to(args.device);b=port(16,16,bias=False).to(args.device);b.load_state_dict(a.state_dict())
        x=torch.randn(2,3,16,device=args.device)
        r1=torch.linalg.qr(torch.randn(16,16,device=args.device)).Q.detach().requires_grad_()
        r2=torch.linalg.qr(torch.randn(8,8,device=args.device)).Q.detach().requires_grad_()
        s1=r1.detach().clone().requires_grad_();s2=r2.detach().clone().requires_grad_()
        y=a(x,r1,r2,transpose);z=b(x,s1,s2,transpose);y.square().mean().backward();z.square().mean().backward()
        error=lambda u,v:float((u-v).norm()/u.norm().clamp_min(1e-30))
        rows.append({'transpose':transpose,'output_relative_error':error(y,z),'r1_gradient_relative_error':error(r1.grad,s1.grad),'r2_gradient_relative_error':error(r2.grad,s2.grad)})
    passed=all(max(r[k] for k in r if k!='transpose')<1e-5 for r in rows)
    result={'source_type':'synthetic_math','status':'PASS_NUMERICAL_ONLY' if passed else 'FAIL','cases':rows,
            'environment':environment_report(),'real_model_quality':'NOT_RUN','implementation_variant':'our FP32 implementation'}
    atomic_json(args.output,result);print(result['status'],rows)
    if not passed:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='third_party/SpinQuant');p.add_argument('--overlay',default='runs/spinquant_overlay_fp32_v2')
    p.add_argument('--device',default='cuda');p.add_argument('--output',default='runs/build_m4_fp32/numerical_report.json');run(p.parse_args())
