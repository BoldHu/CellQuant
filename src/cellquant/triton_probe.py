"""Imported only when Triton is installed, to test an actual target CUDA kernel."""
import triton
import triton.language as tl


@triton.jit
def _add(X,Y,Z,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(Z+i,tl.load(X+i,i<N,other=0)+tl.load(Y+i,i<N,other=0),i<N)


def probe():
    import torch
    x=torch.randn(1025,device='cuda');y=torch.randn_like(x);z=torch.empty_like(x)
    _add[(triton.cdiv(x.numel(),256),)](x,y,z,x.numel(),256)
    torch.cuda.synchronize();error=float((z-(x+y)).abs().max())
    return {'status':'PASS' if error==0 else 'FAIL','max_absolute_error':error,'scope':'Triton vector-add compilation and launch only'}
