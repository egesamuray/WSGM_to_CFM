# improved_diffusion/curvelet_ops.py
import math
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F


def _fftshift2d(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(x, dim=(-2, -1))


def _ifftshift2d(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifftshift(x, dim=(-2, -1))


def _fft2c(x: torch.Tensor) -> torch.Tensor:
    return _fftshift2d(torch.fft.fft2(x, norm="ortho"))


def _ifft2c(X: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(_ifftshift2d(X), norm="ortho").real


def _freq_grid(h: int, w: int, device, dtype):
    fy = torch.fft.fftfreq(h, d=1.0, device=device, dtype=dtype).unsqueeze(1).repeat(1, w)
    fx = torch.fft.fftfreq(w, d=1.0, device=device, dtype=dtype).unsqueeze(0).repeat(h, 1)
    fy = _fftshift2d(fy)
    fx = _fftshift2d(fx)
    return fx, fy


def _raised_cosine(x: torch.Tensor, x0: float, x1: float) -> torch.Tensor:
    out = torch.zeros_like(x)
    left = x <= x0
    right = x >= x1
    mid = (~left) & (~right)
    out[right] = 1.0
    z = (x[mid] - x0) / max(1e-9, (x1 - x0))
    out[mid] = 0.5 - 0.5 * torch.cos(math.pi * z)
    return out


def _make_filters(
    h: int,
    w: int,
    J: int,
    angles_per_scale: Optional[List[int]],
    device,
    dtype,
) -> Tuple[torch.Tensor, List[List[torch.Tensor]], Dict]:
    fx, fy = _freq_grid(h, w, device, dtype)
    r = torch.sqrt(fx * fx + fy * fy)
    rmax = torch.max(r)
    r_norm = r / (rmax + 1e-12)
    theta = torch.atan2(fy, fx)  # [-pi, pi]

    if not angles_per_scale:
        angles_per_scale = [8, 16, 32] if max(h, w) >= 64 else [8, 16]

    base = 0.06 if max(h, w) <= 64 else 0.04
    sigmas = [base * (2 ** j) for j in range(J + 1)]

    coarse = torch.exp(-((r_norm / (sigmas[0] + 1e-9)) ** 4))
    wedge_filts: List[List[torch.Tensor]] = []
    eps = 1e-8

    for j in range(1, J + 1):
        s_lo = sigmas[j - 1]
        s_hi = sigmas[j]
        ring_hi = torch.exp(-((r_norm / (s_hi + 1e-9)) ** 4))
        ring_lo = torch.exp(-((r_norm / (s_lo + 1e-9)) ** 4))
        ring = (ring_lo - ring_hi).clamp(min=0.0)

        Wj = int(angles_per_scale[-j]) if len(angles_per_scale) >= j else angles_per_scale[-1]
        wedges_j: List[torch.Tensor] = []
        ang_hw = math.pi / max(4.0, Wj)
        for k in range(Wj):
            phi = -math.pi + (2 * math.pi) * (k / Wj)
            dtheta = torch.remainder(theta - phi + math.pi, 2 * math.pi) - math.pi
            ang_win = _raised_cosine(dtheta.abs(), 0.0, ang_hw) * _raised_cosine((ang_hw - dtheta.abs()).abs(), 0.0, ang_hw)
            F = (ring * ang_win).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
            wedges_j.append(F + eps)
        wedge_filts.append(wedges_j)

    total = coarse.clone()
    for j in range(J):
        for Fw in wedge_filts[j]:
            total = total + Fw.squeeze(0).squeeze(0)
    total = total + eps
    coarse = (coarse / total).unsqueeze(0).unsqueeze(0)
    for j in range(J):
        for k in range(len(wedge_filts[j])):
            wedge_filts[j][k] = wedge_filts[j][k] / total.unsqueeze(0).unsqueeze(0)

    meta = dict(h=h, w=w, J=J, angles_per_scale=angles_per_scale)
    return coarse, wedge_filts, meta


@lru_cache(maxsize=32)
def _filters_cpu_cached(h: int, w: int, J: int, angles_tuple: tuple):
    """Build filters once on CPU/float32 and cache them."""
    angles = list(angles_tuple) if angles_tuple else None
    coarse_cpu, wedge_cpu, meta = _make_filters(
        h, w, J, angles, device=torch.device("cpu"), dtype=torch.float32
    )
    wedge_cpu = [[f.cpu().float() for f in row] for row in wedge_cpu]
    return coarse_cpu.cpu().float(), wedge_cpu, meta


def _make_filters_fast(h, w, J, angles_per_scale, device, dtype):
    angles_tuple = tuple(int(a) for a in (angles_per_scale or []))
    coarse_cpu, wedge_cpu, meta = _filters_cpu_cached(h, w, J, angles_tuple)
    coarse = coarse_cpu.to(device=device, dtype=dtype)
    wedge_filts = [[f.to(device=device, dtype=dtype) for f in row] for row in wedge_cpu]
    return coarse, wedge_filts, meta


@torch.no_grad()
def fdct2(
    x: torch.Tensor,
    J: Optional[int] = None,
    angles_per_scale: Optional[List[int]] = None,
) -> Dict:
    """
    Curvelet-like forward transform.

    Args:
        x: (B,C,H,W) with C in {1,3}, in [-1,1] or [0,1]
        J: number of oriented scales (1..J), default based on image size
        angles_per_scale: list coarsest->finest, e.g., [8,16,32]
    Returns:
        dict: {
          'coarse': (B,C,Hc,Wc),
          'bands':  list over scales j=1..J of list of (B,C,Hj,Wj) wedge tensors,
          'meta':   {'angles_per_scale': [...], 'color_channels': C, ...}
        }
    """
    assert x.dim() == 4, "x must be (B,C,H,W)"
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    if J is None:
        J = 3 if max(H, W) <= 128 else 4

    X = _fft2c(x)
    # cached filters
    coarse_filt, wedge_filts, meta = _make_filters_fast(H, W, J, angles_per_scale, device, dtype)

    Coarse = X * coarse_filt
    coarse = _ifft2c(Coarse)

    bands: List[List[torch.Tensor]] = []
    for j in range(J):
        wedges_j = []
        for Fw in wedge_filts[j]:
            Wedge = X * Fw
            wspace = _ifft2c(Wedge)
            wedges_j.append(wspace)
        bands.append(wedges_j)

    meta = dict(meta, color_channels=C)
    return {"coarse": coarse, "bands": bands, "meta": meta}


def pack_highfreq(coeffs: Dict, j: int) -> torch.Tensor:
    """
    Stack all wedge subbands at scale j along channel dimension.
    j: 1=finest
    Returns (B, C*W_j, N_j, N_j)
    """
    bands_j = coeffs["bands"][j - 1]
    B, C, Hj, Wj = bands_j[0].shape
    Nj = math.ceil(Hj / 2)  # coarse size at this level
    packed_list = []
    for w in bands_j:
        w_small = F.interpolate(w, size=(Nj, Nj), mode="area")
        packed_list.append(w_small)
    return torch.cat(packed_list, dim=1)


def unpack_highfreq(packed: torch.Tensor, j: int, meta: Dict) -> List[torch.Tensor]:
    """
    Inverse of pack_highfreq.
    packed: (B, C*W_j, N_j, N_j)
    Returns list[W_j] of upsampled wedges (B,C, 2*N_j, 2*N_j)
    """
    B, packed_C, Nj, _ = packed.shape
    angles = meta.get("angles_per_scale", None) or []
    C = meta.get("color_channels", None)
    if C is None:
        if angles and 0 < j <= len(angles):
            Wj = int(angles[-j])
            C = max(1, packed_C // max(1, Wj))
        else:
            C = 3
    if angles and 0 < j <= len(angles):
        Wj = int(angles[-j])
    else:
        Wj = max(1, packed_C // C)

    wedges = []
    for k in range(Wj):
        w_small = packed[:, C * k : C * (k + 1), :, :]
        w_up = F.interpolate(w_small, scale_factor=2, mode="bilinear", align_corners=False)
        wedges.append(w_up)
    return wedges


@torch.no_grad()
def ifdct2(coeffs: Dict, output_size: Optional[int] = None) -> torch.Tensor:
    """
    Approximate inverse: start from 'coarse', then for each oriented scale
    (coarse -> finer), upsample ×2 and add all wedge components.
    """
    img = coeffs["coarse"]
    bands = coeffs.get("bands", [])
    for band in bands:
        img = F.interpolate(img, scale_factor=2, mode="bilinear", align_corners=False)
        high_sum = torch.zeros_like(img)
        for w in band:
            high_sum = high_sum + w
        img = img + high_sum
    if output_size is not None and img.shape[-2] != output_size:
        img = F.interpolate(img, size=(output_size, output_size), mode="bilinear", align_corners=False)
    return img.clamp(-1, 1)
