# scripts/image_sample.py
import os
import numpy as np
from PIL import Image
import torch
from improved_diffusion import script_util
from improved_diffusion import curvelet_datasets, curvelet_ops

def main():
    args = script_util.create_argparser().parse_args()
    script_util.update_model_channels(args)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion = script_util.create_model_and_diffusion(args)
    model.to(dev)
    setattr(model, "device", dev)
    model.eval()

    # Load weights if provided
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    if args.task == "curvelet":
        if args.conditional:
            assert args.cond_dir is not None, "Must specify --cond_dir for conditional sampling"
            cond_paths = sorted(
                os.path.join(args.cond_dir, f)
                for f in os.listdir(args.cond_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            os.makedirs(args.output_dir, exist_ok=True)

            mean, std = curvelet_datasets.curvelet_stats(
                args.j, args.data_dir,
                angles_per_scale=[int(a) for a in args.angles_per_scale.split(',')] if args.angles_per_scale else None
            )
            mean, std = torch.tensor(mean, device=dev), torch.tensor(std, device=dev)

            for img_path in cond_paths:
                cond_img = Image.open(img_path).convert("RGB")
                cond_arr = (np.array(cond_img).astype(np.float32) / 127.5) - 1.0
                cond_t = torch.from_numpy(cond_arr.transpose(2, 0, 1)).float().unsqueeze(0).to(dev)
                # Whiten coarse
                cond_t = (cond_t - mean[:3][None, :, None, None]) / std[:3][None, :, None, None]

                # Sample high-frequency wedges
                shape = (1, args.in_channels, cond_t.shape[-2], cond_t.shape[-1])
                model_kwargs = {"low_res": cond_t}
                sample = diffusion.p_sample_loop(model, shape, model_kwargs=model_kwargs, device=dev)

                # Unwhiten predicted high
                sample = sample * std[3:][None, :, None, None] + mean[3:][None, :, None, None]

                # (Your downstream inverse-curvelet logic remains as you wrote.)

        else:
            # Unconditional coarse generation
            os.makedirs(args.output_dir, exist_ok=True)
            num = args.num_samples
            h = w = args.image_size // (2 ** args.j) if args.image_size else 64 // (2 ** args.j)
            shape = (min(args.batch_size, num), args.in_channels, h, w)

            mean, std = curvelet_datasets.curvelet_stats(args.j, args.data_dir)
            mean, std = torch.tensor(mean[:3], device=dev), torch.tensor(std[:3], device=dev)

            saved = 0
            while saved < num:
                sample = diffusion.p_sample_loop(model, shape, device=dev)
                sample = sample * std[None, :, None, None] + mean[None, :, None, None]
                sample = sample.clamp(-1, 1)
                for i in range(sample.size(0)):
                    img_arr = ((sample[i].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
                    Image.fromarray(img_arr).save(os.path.join(args.output_dir, f"coarse_j{args.j}_{saved+i}.png"))
                saved += sample.size(0)

if __name__ == "__main__":
    main()
