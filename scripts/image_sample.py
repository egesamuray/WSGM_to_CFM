# scripts/image_sample.py
import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch

from improved_diffusion import script_util
from improved_diffusion import curvelet_datasets, curvelet_ops


def _latest_ckpt(d):
    if not d or not os.path.isdir(d):
        return None
    cands = [p for p in glob.glob(os.path.join(d, "*.pt")) if os.path.isfile(p)]
    if not cands:
        return None
    ema = [p for p in cands if "ema" in os.path.basename(p).lower()]
    if ema:
        ema.sort(key=os.path.getmtime, reverse=True)
        return ema[0]
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]


def _angles_list(s):
    if not s:
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _steps_or_default(x, default_val=256):
    try:
        xv = int(x)
        return xv if xv > 0 else default_val
    except Exception:
        return default_val


def main():
    parser = script_util.create_argparser()
    parser.add_argument("--coarse_model_path", type=str, default=None)
    parser.add_argument("--cond_model_path", type=str, default=None)
    args = parser.parse_args()

    assert args.task == "curvelet"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    angles = _angles_list(args.angles_per_scale)
    C = int(args.color_channels)
    W_j = max(1, script_util._wedges_at_scale(args.j if args.j else 1, angles))
    full_size = args.large_size if args.large_size else (args.image_size or 32)
    coarse_hw = max(4, full_size // (2 ** max(1, int(args.j))))

    coarse_model_path = args.coarse_model_path or os.environ.get("OPENAI_LOGDIR_COARSE")
    cond_model_path = args.cond_model_path or os.environ.get("OPENAI_LOGDIR_COND")
    if coarse_model_path and os.path.isdir(coarse_model_path):
        coarse_model_path = _latest_ckpt(coarse_model_path)
    if cond_model_path and os.path.isdir(cond_model_path):
        cond_model_path = _latest_ckpt(cond_model_path)
    if not coarse_model_path or not os.path.isfile(coarse_model_path):
        root = os.path.join("results", f"curvelet_J{args.j}")
        coarse_model_path = _latest_ckpt(os.path.join(root, "coarse"))
    if not cond_model_path or not os.path.isfile(cond_model_path):
        root = os.path.join("results", f"curvelet_J{args.j}")
        cond_model_path = _latest_ckpt(os.path.join(root, "cond"))
    if not coarse_model_path or not cond_model_path:
        raise FileNotFoundError("Could not resolve coarse/conditional checkpoints.")

    outdir = args.output_dir or os.path.join("results", f"curvelet_J{args.j}", "samples")
    os.makedirs(outdir, exist_ok=True)

    steps_for_sampling = _steps_or_default(getattr(args, "diffusion_steps", -1), default_val=256)

    params_coarse = script_util.model_and_diffusion_defaults(task="curvelet")
    params_coarse.update(dict(
        j=args.j, conditional=False, angles_per_scale=angles,
        large_size=full_size, small_size=full_size, diffusion_steps=steps_for_sampling,
        color_channels=C,
    ))
    coarse_model, coarse_diff = script_util.create_model_and_diffusion(
        task="curvelet",
        **script_util.args_to_dict(argparse.Namespace(**params_coarse), params_coarse.keys())
    )
    coarse_model.load_state_dict(torch.load(coarse_model_path, map_location="cpu"))
    coarse_model.to(device).eval()

    params_cond = script_util.model_and_diffusion_defaults(task="curvelet")
    params_cond.update(dict(
        j=args.j, conditional=True, angles_per_scale=angles,
        large_size=full_size, small_size=full_size, diffusion_steps=steps_for_sampling,
        color_channels=C,
    ))
    cond_model, cond_diff = script_util.create_model_and_diffusion(
        task="curvelet",
        **script_util.args_to_dict(argparse.Namespace(**params_cond), params_cond.keys())
    )
    cond_model.load_state_dict(torch.load(cond_model_path, map_location="cpu"))
    cond_model.to(device).eval()

    stats_npz = os.path.join("results", f"curvelet_J{args.j}", f"curvelet_stats_j{args.j}.npz")
    if os.path.isfile(stats_npz):
        npz = np.load(stats_npz)
        mean = torch.from_numpy(npz["mean"]).float().to(device)
        std = torch.from_numpy(npz["std"]).float().clamp_min(1e-6).to(device)
    else:
        mean, std = curvelet_datasets.curvelet_stats(
            j=args.j, dir_name=args.data_dir, angles_per_scale=angles,
            image_size=full_size, limit=min(256, len(curvelet_datasets._list_images(args.data_dir))),
            device="cpu", color_channels=C,
        )
        mean, std = mean.to(device), std.to(device)

    mean_w = mean[C:C + C * W_j].view(1, C * W_j, 1, 1)
    std_w = std[C:C + C * W_j].clamp_min(1e-6).view(1, C * W_j, 1, 1)

    num = int(args.num_samples)
    done = 0
    while done < num:
        bs = min(args.batch_size, num - done)

        coarse_shape = (bs, C, coarse_hw, coarse_hw)
        with torch.no_grad():
            coarse = coarse_diff.p_sample_loop(coarse_model, coarse_shape, device=device)
            coarse = coarse.clamp(-1, 1)

        wedge_shape = (bs, C * W_j, coarse_hw, coarse_hw)
        with torch.no_grad():
            wedges_white = cond_diff.p_sample_loop(
                cond_model, wedge_shape, model_kwargs={"conditional": coarse}, device=device
            )
        wedges = wedges_white * std_w + mean_w

        wedges_list = curvelet_ops.unpack_highfreq(
            wedges, j=args.j, meta={"angles_per_scale": angles or [], "color_channels": C}
        )
        coeffs = {"coarse": coarse, "bands": [wedges_list]}
        final = curvelet_ops.ifdct2(coeffs, output_size=full_size).clamp(-1, 1)

        arr = final.detach().cpu().numpy()
        for i in range(arr.shape[0]):
            if C == 1:
                im = ((arr[i, 0] + 1.0) * 127.5).astype(np.uint8)
                Image.fromarray(im).save(os.path.join(outdir, f"sample_{done + i:06d}.png"))
            else:
                im = ((arr[i].transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
                Image.fromarray(im).save(os.path.join(outdir, f"sample_{done + i:06d}.png"))
        done += bs

    print(f"Saved {num} images to {outdir}")


if __name__ == "__main__":
    main()
