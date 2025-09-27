# scripts/curvelet_generate.py
import argparse, os
import numpy as np
from PIL import Image
import torch

from improved_diffusion import script_util, curvelet_datasets, curvelet_ops

def main():
    parser = argparse.ArgumentParser(description="Generate images with a Curvelet Score-Based Generative Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of the training dataset (for stats)")
    parser.add_argument("--coarse_model", type=str, required=True, help="Path to the trained coarse model checkpoint (.pt)")
    parser.add_argument("--cond_model", type=str, required=True, help="Path to the trained conditional model checkpoint (.pt)")
    parser.add_argument("--j", type=int, required=True, help="Scale index (1 = finest scale)")
    parser.add_argument("--angles_per_scale", type=str, default=None, help="Comma-separated list of wedges per scale (coarsest->finest), e.g. '8,16,32'")
    parser.add_argument("--image_size", type=int, required=True, help="Resolution (height=width) of the full images")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--diffusion_steps", type=int, default=250, help="Number of diffusion timesteps for sampling")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save generated images")
    args = parser.parse_args()

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parse angles_per_scale if provided
    angles = None
    if args.angles_per_scale:
        angles = [int(x.strip()) for x in args.angles_per_scale.split(",") if x.strip()]

    # Load models and diffusion settings
    # Create coarse (unconditional) model
    params_coarse = script_util.model_and_diffusion_defaults(task="curvelet")
    params_coarse.update({
        "j": args.j,
        "conditional": False,
        "angles_per_scale": angles if angles is not None else None,
        "diffusion_steps": args.diffusion_steps,
        "large_size": args.image_size,
        "small_size": args.image_size
    })
    model_coarse, diffusion_coarse = script_util.create_model_and_diffusion(**params_coarse)
    model_coarse.load_state_dict(torch.load(args.coarse_model, map_location="cpu"))
    model_coarse.to(device).eval()
    # Create conditional model
    params_cond = script_util.model_and_diffusion_defaults(task="curvelet")
    params_cond.update({
        "j": args.j,
        "conditional": True,
        "angles_per_scale": angles if angles is not None else None,
        "diffusion_steps": args.diffusion_steps,
        "large_size": args.image_size,
        "small_size": args.image_size
    })
    model_cond, diffusion_cond = script_util.create_model_and_diffusion(**params_cond)
    model_cond.load_state_dict(torch.load(args.cond_model, map_location="cpu"))
    model_cond.to(device).eval()

    # Compute or load dataset statistics for whitening/unwhitening
    mean, std = curvelet_datasets.curvelet_stats(
        j=args.j, dir_name=args.data_dir, angles_per_scale=angles, image_size=args.image_size,
        limit=min(256, len(curvelet_datasets._list_images(args.data_dir))), device="cpu"
    )
    mean, std = mean.to(device), std.to(device)
    # Split mean/std for coarse (first 3 channels) and wedge (remaining) channels
    # (Avoid very small std by clamping)
    coarse_mean = mean[:3].view(1, 3, 1, 1)
    coarse_std  = std[:3].clamp_min(1e-6).view(1, 3, 1, 1)
    wedge_mean  = mean[3:].view(1, -1, 1, 1)  # 3*W_j channels
    wedge_std   = std[3:].clamp_min(1e-6).view(1, -1, 1, 1)

    os.makedirs(args.output_dir, exist_ok=True)
    coarse_h = args.image_size // (2 ** args.j)  # height/width of coarse images
    samples_generated = 0
    # Generate images in batches
    while samples_generated < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - samples_generated)
        # 1. Sample coarse images (unconditional)
        shape = (batch_size, 3, coarse_h, coarse_h)
        with torch.no_grad():
            coarse_batch = diffusion_coarse.p_sample_loop(
                model_coarse, shape, device=device
            )
        # Unwhiten coarse (since model outputs normalized data)
        coarse_batch = coarse_batch * coarse_std + coarse_mean
        coarse_batch = coarse_batch.clamp(-1, 1)
        # 2. Sample high-frequency wedges (conditional on coarse)
        # Whiten coarse for conditional model input
        cond_in = (coarse_batch - coarse_mean) / coarse_std
        wedge_channels = model_cond.in_channels  # = 3 * W_j
        shape = (batch_size, wedge_channels, coarse_h, coarse_h)
        with torch.no_grad():
            wedge_batch_white = diffusion_cond.p_sample_loop(
                model_cond, shape, model_kwargs={"low_res": cond_in}, device=device
            )
        # Unwhiten predicted wedge coefficients
        wedge_batch = wedge_batch_white * wedge_std + wedge_mean
        # 3. Inverse Curvelet transform: combine coarse and wedges to full images
        # Unpack the high-frequency wedges into spatial wedge images
        wedges_list = curvelet_ops.unpack_highfreq(wedge_batch, j=args.j, meta={"angles_per_scale": angles or []})
        coeffs = {"coarse": coarse_batch, "bands": [wedges_list]}
        final_batch = curvelet_ops.ifdct2(coeffs)
        final_batch = final_batch.clamp(-1, 1)  # ensure output range is [-1,1]
        # 4. Save images to output directory
        final_batch_cpu = final_batch.cpu().numpy()
        for i in range(final_batch_cpu.shape[0]):
            # Convert from [-1,1] to [0,255] uint8
            img_arr = ((final_batch_cpu[i].transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
            Image.fromarray(img_arr).save(os.path.join(
                args.output_dir, f"sample_{samples_generated + i}.png"
            ))
        samples_generated += batch_size

if __name__ == "__main__":
    main()
