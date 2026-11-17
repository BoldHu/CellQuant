"""Torch-only FP64 event geometry and polynomial roots (CPU is a test device).

Arrays stay on their input device. Python controls variable-length work and
reads scalar decisions; it never transfers arrays to NumPy/SciPy for arithmetic.
These are numerical roots, not interval-certified solutions.
"""
from __future__ import annotations

import functools
import math
import torch


def scalar(value, like):
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def basis(theta, like):
    t = scalar(theta, like)
    return torch.stack((torch.ones_like(t), t.cos(), t.sin()))


def trig_value(real, theta):
    t = scalar(theta, real)
    k = torch.arange(1, (real.shape[-1]+1)//2, device=real.device, dtype=real.dtype)
    return real[..., 0] + (real[..., 1::2]*torch.cos(t[..., None]*k) +
                          real[..., 2::2]*torch.sin(t[..., None]*k)).sum(-1)


def affine_roots(coeff, left, right):
    z = coeff.reshape(-1, 3)
    r = torch.hypot(z[:, 1], z[:, 2])
    ratio = -z[:, 0]/torch.where(r > 0, r, torch.ones_like(r))
    ids = torch.where((r > 0) & (ratio.abs() <= 1+8*torch.finfo(z.dtype).eps))[0]
    phi = torch.atan2(z[ids, 2], z[ids, 1])
    alpha = ratio[ids].clamp(-1, 1).acos()
    shifts = scalar([-2*math.pi, 0., 2*math.pi], z)
    t = torch.stack((phi-alpha, phi+alpha), -1)[..., None]+shifts
    ix = ids[:, None, None].expand_as(t).reshape(-1);t = t.reshape(-1)
    residual = (z[ix, 0]+z[ix, 1]*t.cos()+z[ix, 2]*t.sin()).abs()
    ok = (t > left) & (t < right) & (residual <= 5e-13*z[ix].abs().sum(1).clamp_min(1e-300))
    return t[ok], ix[ok]


def owner_intervals(coeff, left, right, stable=False, budget=None):
    """Complete envelope. Whole-row support uses GPU vectorized edge walks.

    At each owner, cross it with every signed group member and take the next
    positive crossing. No reduction-channel truncation or CPU convex hull.
    """
    moving = coeff[:, 1:].ne(0).any(1)
    fixed = torch.where(~moving)[0];variable = torch.where(moving)[0]
    if fixed.numel():fixed = fixed[coeff[fixed, 0].abs().argmax()].reshape(1)
    if stable:
        i = fixed[0]
        return [(scalar(left, coeff), scalar(right, coeff), i, coeff[i, 0].sign())]
    if variable.numel() <= 2:
        members = torch.cat((fixed, variable));candidates = [scalar([left, right], coeff)]
        for pos, i in enumerate(members):
            candidates.append(affine_roots(coeff[i], left, right)[0])
            for j in members[pos+1:]:
                candidates.append(affine_roots(torch.stack((coeff[i]-coeff[j], coeff[i]+coeff[j])), left, right)[0])
        cuts = torch.unique(torch.cat(candidates), sorted=True)
    else:
        if bool(coeff[:, 0].ne(0).any()):raise ValueError('Whole-row orbit requires zero constant')
        # Positive scaling does not change the support envelope; normalize to
        # make near-tie checks independent of the magnitude of the weights.
        points = torch.cat((coeff[:, 1:], -coeff[:, 1:]), 0)
        points = points/points.abs().amax().clamp_min(torch.finfo(coeff.dtype).tiny)
        current = scalar(left, coeff);end = scalar(right, coeff);boundaries = [current]
        values=points@torch.stack((current.cos(),current.sin()))
        slope=points@torch.stack((-current.sin(),current.cos()))
        owner=slope.masked_fill(values!=values.max(),-torch.inf).argmax()
        for _ in range(2*len(points)+2):
            if budget is not None:budget.check()
            delta = points-points[owner]
            z = torch.cat((torch.zeros_like(delta[:, :1]), delta), 1)
            roots, ix = affine_roots(z, current, end)
            derivative = -z[ix, 1]*roots.sin()+z[ix, 2]*roots.cos()
            positive=derivative>0
            roots,ix,derivative=roots[positive],ix[positive],derivative[positive]
            if not roots.numel():break
            nxt = roots.min()
            if not bool(nxt > current):raise RuntimeError('Nonadvancing support envelope')
            # Carry the crossing owner to the right interval. Re-selecting
            # argmax AT the rounded boundary can choose the outgoing owner.
            at=roots==nxt
            owner=ix[at][derivative[at].argmax()]
            boundaries.append(nxt);current = nxt
        else:raise RuntimeError('Support envelope did not terminate')
        cuts = torch.unique(torch.stack([*boundaries, end]), sorted=True)
    intervals = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        v = coeff@basis(lo+(hi-lo)/2, coeff);owner = v.abs().argmax();sign = v[owner].sign()
        if intervals and bool(intervals[-1][2] == owner) and bool(intervals[-1][3] == sign):
            intervals[-1] = (intervals[-1][0], hi, owner, sign)
        else:intervals.append((lo, hi, owner, sign))
    return intervals


def square_coeff(residual):
    """Sum of squares of degree-two real trigonometric coefficients."""
    r = residual.reshape(5, -1)
    f = torch.stack(((r[3]+1j*r[4])/2, (r[1]+1j*r[2])/2,
                     r[0].to(torch.complex128), (r[1]-1j*r[2])/2, (r[3]-1j*r[4])/2))
    # A 5x5 Gram contraction, then anti-diagonal sums of the Fourier product.
    gram = f@f.T
    ij = torch.arange(5, device=r.device)
    indices = (ij[:, None]+ij[None, :]).reshape(-1)
    conv = torch.zeros(9, dtype=gram.dtype, device=r.device).scatter_add_(0, indices, gram.reshape(-1))
    result = torch.empty(9, dtype=r.dtype, device=r.device);result[0] = conv[4].real
    result[1::2] = 2*conv[5:].real;result[2::2] = -2*conv[5:].imag
    return result


def product_coeff(a, x):
    return torch.stack((a[0]@x[0]+(a[1]@x[1]+a[2]@x[2])/2,
                        a[0]@x[1]+a[1]@x[0], a[0]@x[2]+a[2]@x[0],
                        (a[1]@x[1]-a[2]@x[2])/2, (a[1]@x[2]+a[2]@x[1])/2))


def scale_square(sc, width):
    a,b,c = sc
    return torch.stack((a*a+(b*b+c*c)/2, 2*a*b, 2*a*c, (b*b-c*c)/2, b*c))*width/4


@functools.lru_cache(maxsize=16)
def polynomial_transform(degree, device):
    # Integer binomial constants are protocol constants, not data-dependent CPU arithmetic.
    matrix = [[0.]*(2*degree+1) for _ in range(2*degree+1)]
    for j in range(degree+1):matrix[2*j][0] = math.comb(degree, j)
    for k in range(1, degree+1):
        for power in range(2*k+1):
            v = math.comb(2*k, power)*(1j**power)
            for j in range(degree-k+1):
                matrix[power+2*j][2*k-1] += v.real*math.comb(degree-k, j)
                matrix[power+2*j][2*k] += v.imag*math.comb(degree-k, j)
    return torch.tensor(matrix, dtype=torch.float64, device=device)


def companion_roots(coeff):
    """Complex shifted QR on the tensor device, avoiding eigvals' small-CPU path.

    PyTorch 2.6 linalg_eig sends small CUDA matrices to CPU LAPACK. Our
    degree <= 8 companions instead use CUDA QR/matmul with numerical deflation.
    Failure to converge is explicit resource blocking, never a CPU fallback.
    """
    from .events import ResourceLimit
    n=len(coeff)-1
    if n==1:return (-coeff[:1]/coeff[1]).to(torch.complex128)
    matrix=torch.zeros((n,n),dtype=torch.complex128,device=coeff.device)
    matrix[1:,:-1]=torch.eye(n-1,dtype=matrix.dtype,device=coeff.device)
    matrix[:,-1]=-coeff[:-1]/coeff[-1]
    roots=[]
    for active in range(n,1,-1):
        a=matrix[:active,:active].clone();eye=torch.eye(active,dtype=a.dtype,device=a.device)
        for iteration in range(256):
            tolerance=32*torch.finfo(coeff.dtype).eps*(a[-1,-1].abs()+a[-2,-2].abs()).clamp_min(1e-300)
            if bool(a[-1,:-1].abs().sum() <= tolerance):break
            u,v,w,z=a[-2,-2],a[-2,-1],a[-1,-2],a[-1,-1]
            disc=torch.sqrt((u-z).square()/4+v*w)
            plus,minus=(u+z)/2+disc,(u+z)/2-disc
            shift=torch.where((plus-z).abs()<(minus-z).abs(),plus,minus)
            # Deterministic exceptional shifts avoid a repeated Wilkinson cycle.
            if iteration and iteration%32==0:shift=z+(1+1j)*a[-1,-2].abs()
            q,r=torch.linalg.qr(a-shift*eye)
            a=r@q+shift*eye
        else:raise ResourceLimit('GPU polynomial QR did not converge; no CPU fallback')
        roots.append(a[-1,-1]);matrix=a[:-1,:-1]
    roots.append(matrix[0,0])
    return torch.stack(roots)


def trig_roots(real, left, right):
    """Half-angle companion roots on the tensor device; actual-quantizer verification is required."""
    degree = (len(real)-1)//2
    coeff = polynomial_transform(degree, str(real.device))@real
    scale = coeff.abs().max()
    if not bool(scale > 0):return real.new_empty(0)
    coeff = coeff/scale
    nonzero = torch.where(coeff.abs() >= 1e-13)[0]
    n = int(nonzero[-1])
    if n == 0:return real.new_empty(0)
    coeff = coeff[:n+1]
    roots = companion_roots(coeff)
    valid = roots.imag.abs() <= 2e-7*roots.real.abs().clamp_min(1)
    theta = 2*roots.real[valid].atan()
    keep = (theta > left+2e-12) & (theta < right-2e-12)
    keep &= trig_value(real, theta).abs() < 2e-7*real.abs().sum().clamp_min(1e-30)
    theta = torch.sort(theta[keep]).values
    if theta.numel() < 2:return theta
    kept=[theta[0]]
    for value in theta[1:]:
        if bool(value-kept[-1]>2e-12):kept.append(value)
    return torch.stack(kept)
