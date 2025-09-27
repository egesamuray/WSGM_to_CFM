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


# Accept standard images and arrays
_FILE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".npy", ".npz")


def _list_images(root: str) -> List[str]:
    root = os.path.expanduser(root)
    paths: List[str] = []
    for ext in _FILE_EXTS:
        paths.extend(glob.glob(os.path.join(root, f"*{ext}")))
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No images found under: {root}")
    return paths


def _make_transform(image_size: Optional[int], color_channels: int) -> T.Compose:
    """
    PIL pipeline for standard image files (not used for npy/npz).
    """
    tfms: List[torch.nn.Module] = []
    if image_size is not None:
        tfms.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC))
    tfms.append(T.ToTensor())  # -> [0,1]
    mean = [0.5] * color_channels
    std = [0.5] * color_channels
    tfms.append(T.Normalize(mean=mean, std=std))  # [-1,1]
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
    idx = -min(j, len(angles_per_scale))
    return int(angles_per_scale[idx])


def _npy_to_tensor(path: str, color_channels: int) -> torch.Tensor:
    """
    Load .npy or .npz -> (C,H,W) float32 in [-1,1].
    - For .npz, uses the first key encountered.
    - If array is (H,W), makes C=1; if C=3 requested, replicates channels.
    - If array is (H,W,1|3) or (1|3,H,W) it is reshaped accordingly.
    - Per-image min-max normalization to [0,1] then scaled to [-1,1].
    """
    if path.lower().endswith(".npz"):
        z = np.load(path)
        if len(z.files) == 0:
            raise ValueError(f"{path} has no arrays.")
        arr = z[z.files[0]]
    else:
        arr = np.load(path)

    arr = np.asarray(arr)

    # Handle shapes
    if arr.ndim == 2:
        H, W = arr.shape
        C = 1
        arr = arr.reshape(H, W, 1)
    elif arr.ndim == 3:
        # (H,W,C) or (C,H,W)
        if arr.shape[0] in (1, 3) and arr.shape[1] != arr.shape[0]:
            arr = np.transpose(arr, (1, 2, 0))
        H, W, C = arr.shape
        if C not in (1, 3):
            if C > 3:
                arr = arr[..., :3]
                C = 3
            else:
                arr = arr[..., :1]
                C = 1
    else:
        raise ValueError(f"Unsupported array shape {arr.shape} for {path}")

    # Per-image robust min-max
    arr = arr.astype(np.float32)
    vmin = np.min(arr)
    vmax = np.max(arr)
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    # Adapt to requested color channels
    if color_channels == 1:
        if C == 3:
            arr = np.mean(arr, axis=-1, keepdims=True)
    else:
        if C == 1:
            arr = np.repeat(arr, 3, axis=-1)

    # (H,W,C) [0,1] -> tensor (C,H,W) [-1,1]
    arr = 2.0 * arr - 1.0
    t = torch.from_numpy(np.transpose(arr, (2, 0, 1)))  # (C,H,W)
    return t


def _load_tensor_from_path(path: str, color_channels: int, image_size: Optional[int]) -> torch.Tensor:
    """
    Return a tensor (C,H,W) in [-1,1], CPU. Works for both images and npy/npz.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".npy", ".npz"):
        x = _npy_to_tensor(path, color_channels)  # (C,H,W)
        if image_size is not None and (x.shape[-2] != image_size or x.shape[-1] != image_size):
            x4 = x.unsqueeze(0)  # (1,C,H,W)
            x4 = torch.nn.functional.interpolate(x4, size=(image_size, image_size), mode="bilinear", align_corners=False)
            x = x4.squeeze(0)
        return x

    # PIL route for standard images
    img = Image.open(path)
    img = img.convert("L") if color_channels == 1 else img.convert("RGB")
    tfm = _make_transform(image_size, color_channels)
    x = tfm(img)  # (C,H,W) in [-1,1]
    return x


@torch.no_grad()
def curvelet_stats(
    j: int,
    dir_name: str,
    angles_per_scale: Optional[Iterable[int] or str] = None,
    image_size: Optional[int] = None,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    color_channels: int = 3,
):
    """
    Compute dataset mean/std over channels of [coarse || packed_wedges_at_scale(j)].
    Returns mean, std (C,), where C = color_channels + color_channels*W_j.
    """
    paths = _list_images(dir_name)
    if limit is not None:
        paths = paths[: int(limit)]

    angles = _angles_parse(angles_per_scale)
    J = len(angles) if angles else max(j, 3)
    W_j = _wedges_at_scale(j, angles)
    _ = device  # signature compatibility

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
        x = _load_tensor_from_path(p, color_channels, image_size).unsqueeze(0)  # (1,C,H,W)

        coeffs = fdct2(x, J=J, angles_per_scale=angles)
        coarse = coeffs["coarse"]                    # (1,C,Hc,Wc)
        packed = pack_highfreq(coeffs, j)            # (1,C*W_j,Nj,Nj)

        if coarse.shape[-2:] != packed.shape[-2:]:
            coarse = torch.nn.functional.interpolate(coarse, size=packed.shape[-2:], mode="bilinear", align_corners=False)

        combo = torch.cat([coarse, packed], dim=1)   # (1, C + C*W_j, Nj, Nj)
        Ctot = combo.size(1)
        combo_flat = combo.view(Ctot, -1)
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
        color_channels: int = 3,
    ):
        super().__init__()
        self.paths = _list_images(image_dir)
        self.image_size = int(image_size) if image_size is not None else None
        self.j = int(j)
        self.conditional = bool(conditional)
        self.angles = _angles_parse(angles_per_scale)
        self.C = int(color_channels)

        if stats is None and self.conditional:
            with torch.no_grad():
                mean, std = curvelet_stats(
                    j=self.j,
                    dir_name=image_dir,
                    angles_per_scale=self.angles,
                    image_size=self.image_size,
                    limit=min(256, len(self.paths)),
                    device="cpu",
                    color_channels=self.C,
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
        x = _load_tensor_from_path(p, self.C, self.image_size).unsqueeze(0)  # (1,C,H,W)

        coeffs = fdct2(x, J=(len(self.angles) if self.angles else max(self.j, 3)), angles_per_scale=self.angles)
        coarse = coeffs["coarse"]          # (1,C,Hc,Wc)
        packed = pack_highfreq(coeffs, self.j)  # (1,C*Wj,Nj,Nj)

        if coarse.shape[-2:] != packed.shape[-2:]:
            coarse = torch.nn.functional.interpolate(coarse, size=packed.shape[-2:], mode="bilinear", align_corners=False)

        if self.conditional:
            X = packed[0]  # (C*Wj,Nj,Nj)
            if self.mean is not None and self.std is not None:
                Wj = packed.size(1) // self.C
                start = self.C
                mean_w = self.mean[start:start + self.C * Wj].view(-1, 1, 1)
                std_w = self.std[start:start + self.C * Wj].view(-1, 1, 1).clamp_min(1e-6)
                X = (X - mean_w) / std_w
            KW = {"conditional": coarse[0]}  # (C,Nj,Nj)
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
    num_workers: int = 0,
    color_channels: int = 3,
):
    """
    Loader with sensible performance defaults:
    - pin_memory = True if CUDA is available
    - num_workers can be overridden by env CURVELET_LOADER_WORKERS (if > num_workers)
    """
    try:
        env_nw = int(os.getenv("CURVELET_LOADER_WORKERS", "0"))
    except Exception:
        env_nw = 0
    nw = num_workers if num_workers > 0 else max(0, env_nw)

    ds = CurveletDataset(
        image_dir=data_dir,
        image_size=image_size,
        j=j,
        conditional=conditional,
        angles_per_scale=angles_per_scale,
        stats=stats,
        color_channels=color_channels,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(nw > 0),
    )

    def _gen():
        while True:
            for X, KW in loader:
                yield X, {k: v for k, v in KW.items()}

    return _gen()
