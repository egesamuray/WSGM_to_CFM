# improved_diffusion/curvelet_datasets.py
import os
import glob
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from .curvelet_ops import fdct2, pack_highfreq

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _list_images(root: str) -> List[str]:
    root = os.path.expanduser(root)
    paths: List[str] = []
    for ext in _IMG_EXTS:
        paths.extend(glob.glob(os.path.join(root, f"*{ext}")))
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No images found under: {root}")
    return paths


def _make_transform(image_size: Optional[int]) -> T.Compose:
    tfms: List[torch.nn.Module] = []
    if image_size is not None:
        tfms.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC))
    tfms.extend([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  # -> [-1,1] on CPU
    return T.Compose(tfms)


def _angles_parse(angles_per_scale: Optional[Iterable[int] or str]) -> Optional[List[int]]:
    if angles_per_scale is None:
        return None
    if isinstance(angles_per_scale, (list, tuple)):
        return [int(x) for x in angles_per_scale]
    s = str(angles_per_scale).strip()
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _wedges_at_scale(j: int, angles_per_scale: Optional[List[int]]) -> int:
    if not angles_per_scale or j is None or j <= 0:
        return 1
    # angles are listed coarsest -> finest; j=1 is finest
    idx = -min(j, len(angles_per_scale))
    return int(angles_per_scale[idx])


@torch.no_grad()
def curvelet_stats(
    j: int,
    dir_name: str,
    angles_per_scale: Optional[Iterable[int] or str] = None,
    image_size: Optional[int] = None,
    limit: Optional[int] = None,
    device: Optional[str] = None,
):
    """
    Compute dataset mean/std over channels of [coarse || packed_wedges_at_scale(j)].
    Returns mean, std (C,), where C = 3 + 3*W_j.  CPU only to avoid CUDA+fork issues.
    """
    paths = _list_images(dir_name)
    if limit is not None:
        paths = paths[: int(limit)]

    angles = _angles_parse(angles_per_scale)
    J = len(angles) if angles else max(j, 3)
    W_j = _wedges_at_scale(j, angles)

    # Force CPU for stability in Colab/num_workers>0 scenarios
    dev = "cpu"

    tfm = _make_transform(image_size)
    sum_c = None
    sumsq_c = None
    count = 0

    def _accumulate(vec_2d: torch.Tensor):
        nonlocal sum_c, sumsq_c, count
        if sum_c is None:
            sum_c = vec_2d.sum(dim=1)
            sumsq_c = (vec_2d ** 2).sum(dim=1)
        else:
            sum_c += vec_2d.sum(dim=1)
            sumsq_c += (vec_2d ** 2).sum(dim=1)
        count += vec_2d.size(1)

    print(f"Computing stats for scale {j} (W_j={W_j}) on {len(paths)} images...")

    for p in paths:
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0)  # (1,3,H,W) CPU

        coeffs = fdct2(x, J=J, angles_per_scale=angles)  # CPU FFT
        coarse = coeffs["coarse"]            # (1,3,Hc,Wc)
        packed = pack_highfreq(coeffs, j)    # (1,3*W_j,Hp,Wp)

        # Align coarse to wedge spatial size
        if coarse.shape[-2:] != packed.shape[-2:]:
            coarse = F.interpolate(coarse, size=packed.shape[-2:], mode="bilinear", align_corners=False)

        combo = torch.cat([coarse, packed], dim=1)  # (1, 3+3*W_j, H, W)
        C = combo.size(1)
        combo_flat = combo.view(C, -1)
        _accumulate(combo_flat)

    assert sum_c is not None and count > 0, "No pixels accumulated for stats."
    mean = sum_c / count
    var = sumsq_c / count - mean ** 2
    std = torch.sqrt(var.clamp_min(1e-12))

    return mean.cpu(), std.cpu()


class CurveletDataset(Dataset):
    """
    If conditional=True: X = packed wedges (whitened), KW = {'conditional': coarse}
    else               : X = coarse, KW = {}
    All CPU; training loop moves tensors to device.
    """
    def __init__(
        self,
        image_dir: str,
        image_size: Optional[int] = None,
        j: int = 1,
        conditional: bool = True,
        angles_per_scale: Optional[Iterable[int] or str] = None,
        stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        super().__init__()
        self.paths = _list_images(image_dir)
        self.image_size = int(image_size) if image_size is not None else None
        self.j = int(j)
        self.conditional = bool(conditional)
        self.angles = _angles_parse(angles_per_scale)
        self.tfm = _make_transform(self.image_size)

        # If stats not provided and conditional=True, compute a small-sample estimate on CPU.
        if stats is None and self.conditional:
            with torch.no_grad():
                mean, std = curvelet_stats(
                    j=self.j,
                    dir_name=image_dir,
                    angles_per_scale=self.angles,
                    image_size=self.image_size,
                    limit=min(256, len(self.paths)),
                    device="cpu",
                )
        elif stats is not None:
            mean, std = stats
        else:
            mean, std = None, None
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.tfm(img).unsqueeze(0)  # (1,3,H,W) CPU

        coeffs = fdct2(x, J=(len(self.angles) if self.angles else max(self.j, 3)), angles_per_scale=self.angles)
        coarse = coeffs["coarse"]              # (1,3,Hc,Wc)
        packed = pack_highfreq(coeffs, self.j) # (1,3*W_j,Hp,Wp)

        if coarse.shape[-2:] != packed.shape[-2:]:
            coarse = F.interpolate(coarse, size=packed.shape[-2:], mode="bilinear", align_corners=False)

        if self.conditional:
            X = packed[0]  # (3*W_j,H,W)
            if self.mean is not None and self.std is not None:
                Wj = packed.size(1) // 3
                start = 3
                mean_w = self.mean[start:start + 3 * Wj].view(-1, 1, 1)
                std_w = self.std[start:start + 3 * Wj].view(-1, 1, 1).clamp_min(1e-6)
                X = (X - mean_w) / std_w
            KW = {"conditional": coarse[0]}  # matches wavelet training kwargs pattern
        else:
            X = coarse[0]
            KW = {}

        return X, KW


def load_data_curvelet(
    data_dir: str,
    batch_size: int,
    j: int,
    conditional: bool,
    image_size: Optional[int] = None,
    angles_per_scale: Optional[Iterable[int] or str] = None,
    stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    deterministic: bool = False,
    num_workers: int = 0,  # IMPORTANT: no forking (fixes CUDA re-init error)
):
    ds = CurveletDataset(
        image_dir=data_dir,
        image_size=image_size,
        j=j,
        conditional=conditional,
        angles_per_scale=angles_per_scale,
        stats=stats,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_workers,
        pin_memory=False,  # keep CPU path simple
        drop_last=True,
    )

    def _gen():
        while True:
            for X, KW in loader:
                # TrainLoop will move tensors/dicts to device.
                yield X, {k: v for k, v in KW.items()}

    return _gen()

