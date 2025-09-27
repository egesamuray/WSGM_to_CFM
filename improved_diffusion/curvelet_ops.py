# improved_diffusion/curvelet_ops.py
import torch
import math

# Try to import external curvelet libraries
try:
    # Option 1: PyLops Curvelab (curvelops)
    from curvelops import FDCT2D
    _has_curvelops = True
except ImportError:
    _has_curvelops = False
try:
    # Option 2: PyCurvelab (if exists)
    import pycurvelab
    _has_pycurvelab = True
except ImportError:
    _has_pycurvelab = False
try:
    # Option 3: Pure Python "curvelets" library
    import curvelets as _curvelets_lib
    _has_curvelets_lib = True
except ImportError:
    _has_curvelets_lib = False

def fdct2(x, J=None, angles_per_scale=None):
    """
    Forward 2D Curvelet transform for a batch of images.
    Args:
      x: Tensor (B,3,H,W) in [-1,1].
      J: number of scales (excluding the final lowpass). If None, choose based on image size.
      angles_per_scale: list of length J with number of wedges per scale (coarsest -> finest).
    Returns:
      Dict with 'coarse': Tensor (B,3,N_J,N_J) low-frequency image,
                   'bands': list of length J of lists of wedge tensors,
                   'meta': transform metadata.
    """
    B, C, H, W = x.shape
    assert C == 3, "Input must have 3 color channels"
    # Determine J and default angles per scale if not provided
    if J is None:
        # Choose J such that final coarse image is >= 4x4
        J = max(int(math.log2(H) - 2), 1)
    if angles_per_scale is None:
        angles_per_scale = []
        ang = 8
        for j in range(1, J+1):
            angles_per_scale.append(min(ang, 32))
            if ang < 32:
                ang *= 2
        # e.g., J=4 -> angles_per_scale = [8, 16, 32, 32]
    # (External curvelet library usage is omitted – using internal implementation)
    angles_rev = list(reversed(angles_per_scale))  # use fine-to-coarse order
    def _fdct2_recursive(img, angle_idx=0):
        # Get current image size
        _, _, Hc, Wc = img.shape
        # Base case: stop if no more scales or image too small
        if angle_idx >= len(angles_rev) or Hc < 2 or Wc < 2:
            return img, []
        # Compute FFT of current image
        device = img.device
        F_img = torch.fft.fftshift(torch.fft.fft2(img, dim=(-2, -1)), dim=(-2, -1))
        # Frequency grid
        if Hc % 2 == 0:
            fy = torch.arange(-Hc//2, Hc//2, device=device)
        else:
            fy = torch.arange(-(Hc//2), Hc//2 + 1, device=device)
        if Wc % 2 == 0:
            fx = torch.arange(-Wc//2, Wc//2, device=device)
        else:
            fx = torch.arange(-(Wc//2), Wc//2 + 1, device=device)
        Vy, Ux = torch.meshgrid(fy, fx, indexing='ij')
        R = torch.sqrt(Ux**2 + Vy**2)
        # Radial cutoff at half the Nyquist frequency (allow downsampling by 2)
        r_cut = math.floor(min(Hc, Wc) / 4.0)
        if r_cut < 1:
            return img, []
        # Separate low frequencies (coarse) and high frequencies
        coarse_mask = (R <= r_cut)
        high_mask   = (R > r_cut)
        F_coarse = F_img * coarse_mask
        F_high   = F_img * high_mask
        # Inverse FFT to spatial domain for coarse part, then downsample by 2
        coarse_full = torch.fft.ifft2(torch.fft.ifftshift(F_coarse, dim=(-2, -1)), dim=(-2, -1)).real
        coarse_small = coarse_full[..., ::2, ::2]  # downsampled coarse image
        # Angular wedge decomposition of high frequencies
        W_cur = angles_rev[angle_idx]             # number of wedges at this scale
        theta = torch.atan2(Vy, Ux).abs()         # angle (0 to pi) for each frequency
        wedge_list = []
        for k in range(W_cur):
            theta_min = k * math.pi / W_cur
            theta_max = (k + 1) * math.pi / W_cur
            if k < W_cur - 1:
                mask = high_mask & (theta >= theta_min) & (theta < theta_max)
            else:
                # Include upper bound for last wedge
                mask = high_mask & (theta >= theta_min) & (theta <= theta_max)
            if not mask.any():
                wedge_image = torch.zeros_like(img)
            else:
                F_wedge = F_img * mask
                wedge_image = torch.fft.ifft2(torch.fft.ifftshift(F_wedge, dim=(-2, -1)), dim=(-2, -1)).real
            wedge_list.append(wedge_image)  # shape (B,3,Hc,Wc)
        # Recurse on the downsampled coarse image for further scales
        coarse_rec, bands_rec = _fdct2_recursive(coarse_small, angle_idx + 1)
        return coarse_rec, bands_rec + [wedge_list]
    # Process each image in the batch (to conserve memory)
    batch_coarse = []
    batch_bands = None
    for b in range(B):
        coarse_b, bands_b = _fdct2_recursive(x[b:b+1], angle_idx=0)
        batch_coarse.append(coarse_b)
        if batch_bands is None:
            # Initialize bands structure with the first image's bands
            batch_bands = [[wedge for wedge in band] for band in bands_b]
        else:
            # Concatenate subsequent images' wedges along batch dimension
            for si, band in enumerate(bands_b):
                for wi, wedge in enumerate(band):
                    batch_bands[si][wi] = torch.cat([batch_bands[si][wi], wedge], dim=0)
    coarse_batch = torch.cat(batch_coarse, dim=0)  # (B,3,N_J,N_J) low-frequency images
    return {
        'coarse': coarse_batch,
        'bands': batch_bands,
        'meta': {'angles_per_scale': angles_per_scale, 'input_shape': (B, C, H, W)}
    }


def ifdct2(coeffs, output_size=None):
    """
    Inverse Curvelet transform. Reconstructs images from coeffs (output of fdct2).
    If output_size is given, upsamples to that size (useful for partial reconstructions).
    """
    coarse = coeffs['coarse']  # (B,3,h,w)
    bands = coeffs.get('bands', [])
    B, C, Hc, Wc = coarse.shape
    # Determine number of scales
    J = len(bands)
    # Starting from finest scale (bands[-1]) to coarsest (bands[0]), iteratively upsample and add
    img = coarse
    # Note: bands[0] corresponds to coarsest oriented band, bands[-1] finest
    for band in bands:
        # band is a list of wedge tensors for this scale, shape (B,3,h',w') each
        # Upsample current image by 2
        img = torch.nn.functional.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
        # Sum all wedge components (they are already at the same resolution as img after upsampling)
        # If wedge components have the same spatial shape as img:
        # Ensure shapes align:
        assert band[0].shape[-2:] == img.shape[-2:], "Wedge shape does not match upsampled coarse shape"
        # Sum across all wedges
        high_sum = torch.zeros_like(img)
        for wedge in band:
            high_sum += wedge
        img = img + high_sum
    # If a specific output size is requested (and different), further interpolate
    if output_size is not None and img.shape[-2:] != (output_size, output_size):
        img = torch.nn.functional.interpolate(img, size=(output_size, output_size), mode='bilinear', align_corners=False)
    return img

def pack_highfreq(coeffs, j):
    """
    Pack the high-frequency wedge coefficients at scale j into a single tensor.
    Returns tensor shape (B, 3*W_j, N_j, N_j), where N_j = size of coarse at scale j.
    """
    bands = coeffs['bands']
    angles_per_scale = coeffs['meta']['angles_per_scale']
    J = len(bands)
    # Find index in bands list: bands[0] is coarsest scale (scale J), bands[-1] is scale1.
    # If input j corresponds to scale j (with coarse size N_j), that should be band index (J-j).
    band_index = J - j
    if band_index < 0 or band_index >= len(bands):
        raise ValueError(f"Invalid scale j={j} for available bands.")
    wedge_list = bands[band_index]  # list of wedge tensors at that scale
    B, C, Hw, Ww = wedge_list[0].shape  # wedge spatial shape (H_wedge, W_wedge)
    # Coarse at scale j shape:
    Nc = math.ceil(Hw / 2)  # since wedge resolution is 2*Nc
    # Downsample each wedge to coarse size (N_j)
    packed_list = []
    for wedge in wedge_list:
        # wedge shape (B,3,Hw,Ww)
        # Downsample by factor=2 using area interpolation (to avoid aliasing)
        wedge_small = torch.nn.functional.interpolate(wedge, size=(Nc, Nc), mode='area')
        packed_list.append(wedge_small)
    # Concatenate wedges along channel dimension
    packed = torch.cat([w for w in packed_list], dim=1)  # (B, 3*W_j, Nc, Nc)
    return packed

def unpack_highfreq(packed, j, meta):
    """
    Unpack a packed wedge tensor into list of wedge images (upsampled to original wedge size).
    `packed`: Tensor (B, 3*W_j, N_j, N_j) packed high-frequency coeffs.
    Returns list of wedge tensors each shape (B,3, 2*N_j, 2*N_j).
    """
    B, packed_C, Nc, Nc2 = packed.shape
    assert Nc == Nc2, "Packed highfreq tensor must be square"
    W_j = packed_C // 3
    angles_per_scale = meta.get('angles_per_scale', None)
    # Each wedge chunk = 3 channels
    wedge_list_small = [ packed[:, 3*k:3*(k+1), :, :] for k in range(W_j) ]
    wedge_list = []
    # Upsample each small wedge to double size
    for wedge_small in wedge_list_small:
        wedge_up = torch.nn.functional.interpolate(wedge_small, scale_factor=2, mode='bilinear', align_corners=False)
        wedge_list.append(wedge_up)
    return wedge_list
