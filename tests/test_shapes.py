# tests/test_shapes.py
import torch
from improved_diffusion import script_util, curvelet_datasets

def test_model_channels():
    # Curvelet conditional model channels
    class Args: pass
    args = Args()
    args.task = "curvelet"
    args.j = 2
    args.angles_per_scale = "8,8"
    args.conditional = True
    script_util.update_model_channels(args)
    # 8 wedges at scale -> in_channels = 24, cond 3
    assert args.in_channels == 24 and args.conditioning_channels == 3 and args.out_channels == 24
    # Unconditional coarse
    args2 = Args()
    args2.task = "curvelet"; args2.j = 3
    args2.conditional = False
    script_util.update_model_channels(args2)
    assert args2.in_channels == 3 and args2.conditioning_channels == 0

def test_dataset_output_shapes():
    # Prepare a tiny synthetic dataset of one image
    import os
    data_dir = "temp_test_images"
    os.makedirs(data_dir, exist_ok=True)
    from PIL import Image
    test_img = Image.fromarray(((np.random.rand(32,32,3)*255).astype('uint8')))
    test_img.save(os.path.join(data_dir, "test.png"))
    # Conditional dataset
    ds_cond = curvelet_datasets.CurveletDataset(data_dir, j=1, conditional=True, angles_per_scale=[8])
    sample = ds_cond[0]  # (C_total, N, N)
    C_total = sample.shape[0]
    # C_total should be 3 (coarse) + 3*W1 (high)
    assert C_total == 3 + 3*8
    # Unconditional dataset
    ds_uncond = curvelet_datasets.CurveletDataset(data_dir, j=1, conditional=False, angles_per_scale=[8])
    sample2 = ds_uncond[0]
    assert sample2.shape[0] == 3  # only coarse channels
