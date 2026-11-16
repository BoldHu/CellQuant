"""Explicit symmetric RTN contract shared by the production solvers.

Safe division uses one for an all-zero group; its physical scale and risk are
zero. Codes and dequantized values are deliberately separate return values.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class QuantSpec:
    bits: int = 4
    group_size: int = -1
    symmetric: bool = True
    clip_ratio: float = 1.0

    def width(self, dimension):
        if not self.symmetric or self.clip_ratio != 1.0:
            raise NotImplementedError("Event solver supports symmetric AbsMax without clipping only")
        if not 2 <= self.bits <= 8:
            raise ValueError("bits must be in [2, 8]")
        g = dimension if self.group_size == -1 else self.group_size
        if not isinstance(g, int) or g <= 0 or dimension % g:
            raise ValueError("Quantization groups must divide the complete reduction dimension")
        return g

    @property
    def qmax(self):
        return 2 ** (self.bits - 1) - 1

    @property
    def qmin(self):
        return -(2 ** (self.bits - 1))


def quantize(values, spec=QuantSpec(), *, scales=None):
    values = np.asarray(values)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected finite, nonempty row-major vectors")
    g = spec.width(values.shape[-1])
    grouped = values.reshape(values.shape[0], -1, g)
    scale = np.max(np.abs(grouped), axis=-1, keepdims=True) / spec.qmax if scales is None else np.asarray(scales)
    if scale.shape != grouped.shape[:-1] + (1,) or (scale < 0).any() or not np.isfinite(scale).all():
        raise ValueError("Invalid physical scales")
    code = np.clip(np.rint(grouped / np.where(scale > 0, scale, 1)), spec.qmin, spec.qmax).astype(np.int16)
    return (code * scale).reshape(values.shape), code.reshape(values.shape), scale


def risk(scales, group_width):
    return float(group_width * np.square(scales).sum(dtype=np.float64) / 4)


def torch_quantize(values, spec=QuantSpec(), *, ste=False, scales=None):
    """Same row-wise contract on CPU/CUDA; STE keeps the scale derivative."""
    import torch
    g = spec.width(values.shape[-1])
    z = values.reshape(*values.shape[:-1], -1, g)
    s = z.abs().amax(dim=-1, keepdim=True) / spec.qmax if scales is None else scales
    v = z / torch.where(s > 0, s, torch.ones_like(s))
    rounded = v + (v.round() - v).detach() if ste else v.round()
    codes = rounded.clamp(spec.qmin, spec.qmax)
    return (codes * s).reshape_as(values), codes.reshape_as(values), s
