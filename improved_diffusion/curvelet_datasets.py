# improved_diffusion/curvelet_datasets.py
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from improved_diffusion import curvelet_ops

# Utility: compute or load per-channel mean and std for given scale
def curvelet_stats(j, data_dir, angles_per_scale=None):
    stats_file = os.path.join(data_dir, f"curvelet_stats_j{j}.npz")
    if os.path.exists(stats_file):
        data = np.load(stats_file)
        mean = data['mean']; std = data['std']
        return mean, std
    # Otherwise, compute stats by iterating dataset images
    # We assume images in data_dir (png/jpg files).
    import glob
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*")))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {data_dir}")
    sum_channels = None
    sum_squares = None
    n_pixels = 0
    # For each image, get curvelet coefficients at scale j
    for path in image_paths:
        from PIL import Image
        img = Image.open(path).convert('RGB')
        img = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1,1]
        img_tensor = torch.from_numpy(img.transpose((2,0,1))).unsqueeze(0)  # (1,3,H,W)
        # Compute coefficients
        H, W = img_tensor.shape[-2:]
        # Determine J as max scale available
        J_max = math.floor(math.log2(min(H, W)))
        if j > J_max:
            raise ValueError(f"Requested scale j={j} but image too small for that many scales.")
        coeffs = curvelet_ops.fdct2(img_tensor, J= j if angles_per_scale is None else len(angles_per_scale), angles_per_scale=angles_per_scale)
        # Pack coarse+high at scale j
        if j == coeffs['meta']['angles_per_scale'].__len__():  # if j corresponds to coarsest scale
            # For coarsest scale, "highfreq" is none; just coarse
            packed = coeffs['coarse']
        else:
            packed = curvelet_ops.pack_highfreq(coeffs, j)
            # Prepend coarse as condition channels
            coarse = coeffs['coarse']
            # Find coarse at that scale j (coarse_j is actually coeffs['coarse'] if j==J, else need to get intermediate coarse?)
            # Here assume if j<max, then we need to get coarse_j by partial inverse? Simplify by computing full transform to get exact coarse_j:
            # Actually fdct2 returns only final coarse (scale J). For j< J, coarse_j is accessible via iterative transform sequence, but not directly returned.
            # We can get coarse_j by reconstructing up to that scale:
            # Reconstruct coarse_{j} by using coarse_final and adding all bands from final up to j+1:
            if j < coeffs['meta']['angles_per_scale'].__len__():
                # Partial inverse: add back bands beyond scale j
                full_img = curvelet_ops.ifdct2(coeffs)
                # Now redo transform up to scale j only
                coeffs_j = curvelet_ops.fdct2(full_img, J=j, angles_per_scale=coeffs['meta']['angles_per_scale'][-j:])
                coarse_j = coeffs_j['coarse']
            else:
                coarse_j = coeffs['coarse']
            packed = torch.cat([coarse_j, packed], dim=1)
        arr = packed.squeeze(0).cpu().numpy()  # shape (C, N_j, N_j)
        C_tot = arr.shape[0]
        # Flatten spatial and accumulate
        if sum_channels is None:
            sum_channels = np.zeros(C_tot, dtype=np.float64)
            sum_squares = np.zeros(C_tot, dtype=np.float64)
        # mean over spatial dims
        arr_flat = arr.reshape(C_tot, -1)
        sum_channels += arr_flat.sum(axis=1)
        sum_squares += (arr_flat**2).sum(axis=1)
        n_pixels += arr_flat.shape[1]
    mean = sum_channels / n_pixels
    var = sum_squares / n_pixels - mean**2
    std = np.sqrt(np.maximum(var, 1e-8))
    np.savez(stats_file, mean=mean, std=std)
    return mean, std

class CurveletDataset(Dataset):
    def __init__(self, data_dir, j, conditional=True, angles_per_scale=None):
        self.data_dir = data_dir
        self.j = j
        self.conditional = conditional
        self.angles_per_scale = angles_per_scale
        # Load or compute stats
        self.mean, self.std = curvelet_stats(j, data_dir, angles_per_scale)
        # List image files
        import glob
        self.files = sorted(glob.glob(os.path.join(data_dir, "*")))
        # Precompute all coefficients and store in memory or disk shards for performance
        # Here, for simplicity, we will compute on the fly in __getitem__ (can be cached externally).
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        from PIL import Image
        img = Image.open(path).convert('RGB')
        img = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1,1] normalized
        img_tensor = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)  # (1,3,H,W)
        # Compute full curvelet coeffs
        coeffs = curvelet_ops.fdct2(img_tensor, J=self.j if self.angles_per_scale is None else len(self.angles_per_scale), angles_per_scale=self.angles_per_scale)
        if self.conditional:
            # Get coarse image at scale j and packed high-freq wedges
            if self.j == len(coeffs['bands']):
                # If j is coarsest scale, no high-freq (this is odd case - unconditional scenario normally)
                coarse = coeffs['coarse']
                high = None
            else:
                # Pack high-frequency at scale j
                high = curvelet_ops.pack_highfreq(coeffs, self.j)
                # Extract coarse_j: reconstruct coarse_j (the conditional input image)
                if self.j < len(coeffs['bands']):
                    # Partial recon to get coarse_j
                    # We can reconstruct the image up to scale j (which yields coarse_{j})
                    # Instead, simpler: take the next band (j+1) coarse directly from transform.
                    # Actually not directly in coeffs, so we do partial ifdct:
                    img_up_to_j = curvelet_ops.ifdct2({'coarse': coeffs['coarse'], 'bands': coeffs['bands'][:len(coeffs['bands'])- (self.j)]})
                    # Now downsample that image to needed coarse size
                    # Actually img_up_to_j is coarse_{j} as image
                    coarse_size = high.shape[-1]  # N_j
                    coarse_img = torch.nn.functional.interpolate(img_up_to_j, size=(coarse_size, coarse_size), mode='area')
                    coarse = coarse_img
                else:
                    coarse = coeffs['coarse']
            # Concatenate coarse (3 ch) and high (3*W_j ch) along channel
            if high is not None:
                combined = torch.cat([coarse, high], dim=1)  # shape (1, 3+3*W_j, N_j, N_j)
            else:
                combined = coarse
            # Whiten channels
            combined_np = combined.squeeze(0).cpu().numpy()
            # Apply channel-wise normalization
            combined_np = (combined_np - self.mean[:, None, None]) / self.std[:, None, None]
            combined_tensor = torch.from_numpy(combined_np).float()
            return combined_tensor
        else:
            # Unconditional: just coarse_J image
            coarse = coeffs['coarse']  # (1,3,N_J,N_J)
            coarse_np = coarse.squeeze(0).cpu().numpy()
            coarse_np = (coarse_np - self.mean[:3, None, None]) / self.std[:3, None, None]  # first 3 channels are coarse
            coarse_tensor = torch.from_numpy(coarse_np).float()
            return coarse_tensor

def load_data_curvelet(data_dir, batch_size, j, conditional=True, angles_per_scale=None, deterministic=False):
    """
    Returns a PyTorch DataLoader or generator that yields curvelet coefficient data.
    If conditional=True, yields (high_coeff_tensor, coarse_cond_tensor) for scale j.
    If conditional=False, yields coarse images for scale j.
    """
    dataset = CurveletDataset(data_dir, j, conditional=conditional, angles_per_scale=angles_per_scale)
    # We can use a DataLoader for deterministic order or infinite generator as needed.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=not deterministic, drop_last=True)
    if conditional:
        # Wrap to yield tuple (high, coarse) splitting channels
        for batch in loader:
            # batch shape (batch_size, C_total, N_j, N_j)
            C_total = batch.shape[1]
            coarse_channels = 3
            high_channels = C_total - 3
            coarse_batch = batch[:, 0:3, ...]
            high_batch = batch[:, 3:, ...]
            yield high_batch, coarse_batch
    else:
        for batch in loader:
            yield batch
