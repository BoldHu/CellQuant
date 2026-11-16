"""Explicit FHT backend for the audited SpinQuant overlay.

CELLQUANT_FHT_BACKEND=torch_reference selects a slow differentiable torch
implementation for compatibility/tests. It never counts as an extension test.
The default requires the real installed fast-hadamard-transform extension.
"""
import os


def torch_hadamard(x, scale=1.0):
    import torch
    n=x.shape[-1]
    if n<1 or n & (n-1):raise ValueError("Power-of-two Hadamard axis required")
    y=x
    step=1
    while step<n:
        z=y.reshape(*x.shape[:-1],-1,2,step)
        y=torch.stack((z[...,0,:]+z[...,1,:],z[...,0,:]-z[...,1,:]),dim=-2).reshape_as(x)
        step*=2
    return y*scale


def hadamard_transform(x, scale=1.0):
    backend=os.environ.get("CELLQUANT_FHT_BACKEND","extension")
    if backend=="torch_reference":return torch_hadamard(x,scale)
    if backend!="extension":raise ValueError("Unknown CELLQUANT_FHT_BACKEND")
    from fast_hadamard_transform import hadamard_transform as extension
    return extension(x,scale=scale)
