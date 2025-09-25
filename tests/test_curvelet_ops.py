# tests/test_curvelet_ops.py
import torch
from improved_diffusion import curvelet_ops
import numpy as np

def test_round_trip_small():
    # Test forward and inverse on a small image (8x8)
    torch.manual_seed(0)
    x = torch.randn(1, 3, 8, 8)  # random image
    coeffs = curvelet_ops.fdct2(x, J=2)  # 2-scale transform (coarse 1 -> 2)
    recon = curvelet_ops.ifdct2(coeffs)
    diff = (recon - x).abs().max().item()
    print("Max reconstruction error:", diff)
    assert diff < 1e-6

def test_wedge_packing_shapes():
    # Test that pack/unpack produce correct shapes
    B = 2
    x = torch.zeros(B, 3, 16, 16)
    coeffs = curvelet_ops.fdct2(x, J=2, angles_per_scale=[8, 8])  # 2 scales, 8 wedges each for simplicity
    # Scale 2 (coarsest oriented band, coarse size 4x4)
    packed2 = curvelet_ops.pack_highfreq(coeffs, j=2)
    assert packed2.shape[1] == 3*8  # 8 wedges * 3
    assert packed2.shape[-1] == coeffs['coarse'].shape[-1]  # coarse2 size 4
    # Scale 1 (finest band, coarse size 8x8)
    packed1 = curvelet_ops.pack_highfreq(coeffs, j=1)
    assert packed1.shape[1] == 3*8
    # Unpack and invert
    wedge_list = curvelet_ops.unpack_highfreq(packed1, 1, coeffs['meta'])
    # After unpack, each wedge should be double coarse1 size (16x16), matching original image res
    for wedge in wedge_list:
        assert wedge.shape[-1] == 16
    # Ensure adding up coarse+wedges yields original (since we had zeros input)
    coarse1 = coeffs['coarse']  # coarse at final scale (2)
    # Compose manually: upsample coarse and add band2, then upsample and add band1
    img = coarse1
    for band in coeffs['bands']:
        img = torch.nn.functional.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
        sum_band = torch.zeros_like(img)
        for wedge in band:
            # align shapes
            if wedge.shape[-1] != img.shape[-1]:
                wedge_up = torch.nn.functional.interpolate(wedge, size=img.shape[-2:], mode='bilinear', align_corners=False)
            else:
                wedge_up = wedge
            sum_band += wedge_up
        img = img + sum_band
    # img should equal original x (which was zero, so result should be ~zero)
    assert torch.allclose(img, x, atol=1e-6)

def test_dtype_device_parity():
    # Test that transform works on CPU vs CUDA and preserves dtype
    img = torch.randn(1, 3, 32, 32).float()
    coeffs_cpu = curvelet_ops.fdct2(img, J=3)
    recon_cpu = curvelet_ops.ifdct2(coeffs_cpu)
    if torch.cuda.is_available():
        coeffs_gpu = curvelet_ops.fdct2(img.cuda(), J=3)
        recon_gpu = curvelet_ops.ifdct2(coeffs_gpu)
        err = (recon_gpu.cpu() - recon_cpu).abs().mean().item()
        print("CPU vs GPU recon diff:", err)
        assert err < 1e-6
    # Check types
    assert recon_cpu.dtype == torch.float32
    assert torch.allclose(recon_cpu, recon_cpu)  # trivial self-check for no NaNs
