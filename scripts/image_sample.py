# scripts/image_sample.py
import os
import torch
from improved_diffusion import script_util
from improved_diffusion import curvelet_datasets, curvelet_ops

def main():
    args = script_util.create_argparser().parse_args()
    script_util.update_model_channels(args)
    model, diffusion = script_util.create_model_and_diffusion(args)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Load model weights
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    # Determine output shape
    if args.task == "curvelet":
        # Conditional or unconditional
        if args.conditional:
            # High-frequency generation
            # Need coarse conditioning images
            assert args.cond_dir is not None, "Must specify --cond_dir for conditional sampling"
            # Load coarse images
            cond_paths = sorted([os.path.join(args.cond_dir, f) for f in os.listdir(args.cond_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
            os.makedirs(args.output_dir, exist_ok=True)
            # Get stats for unwhitening
            mean, std = curvelet_datasets.curvelet_stats(args.j, args.data_dir, angles_per_scale=[int(a) for a in args.angles_per_scale.split(',')] if args.angles_per_scale else None)
            mean, std = torch.tensor(mean), torch.tensor(std)
            for img_path in cond_paths:
                from PIL import Image
                cond_img = Image.open(img_path).convert('RGB')
                cond_arr = (np.array(cond_img).astype(np.float32) / 127.5) - 1.0
                cond_t = torch.from_numpy(cond_arr.transpose(2,0,1)).float().unsqueeze(0)
                # Whiten coarse
                cond_t = (cond_t - mean[:3][None,:,None,None]) / std[:3][None,:,None,None]
                cond_t = cond_t.to(model.device)
                # Sample high-frequency wedges
                shape = (1, args.in_channels, cond_t.shape[-2], cond_t.shape[-1])
                model_kwargs = {"low_res": cond_t}
                sample = diffusion.p_sample_loop(model, shape, model_kwargs=model_kwargs, device=model.device)
                # Unwhiten predicted high
                sample = sample * std[3:][None,:,None,None] + mean[3:][None,:,None,None]
                # Combine with coarse and inverse transform to get next image
                packed = torch.cat([cond_t * std[:3][None,:,None,None] + mean[:3][None,:,None,None], sample], dim=1)
                # Unpack and inverse curvelet for one level
                # Unpack wedge list
                wedge_list = curvelet_ops.unpack_highfreq(sample, args.j, {"angles_per_scale": [int(a) for a in args.angles_per_scale.split(',')] if args.angles_per_scale else None})
                # Upsample coarse
                coarse = (cond_t * std[:3][None,:,None,None] + mean[:3][None,:,None,None]).to(model.device)
                coarse_up = torch.nn.functional.interpolate(coarse, scale_factor=2, mode='bilinear', align_corners=False)
                combined = coarse_up
                for wedge in wedge_list:
                    combined += wedge.to(model.device)
                recon_img = combined.clamp(-1, 1)
                # Save output image
                out_arr = ((recon_img.squeeze(0).cpu().numpy().transpose(1,2,0) + 1) * 127.5).astype(np.uint8)
                out_img = Image.fromarray(out_arr)
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                out_img.save(os.path.join(args.output_dir, f"{base_name}_scale{args.j}.png"))
        else:
            # Unconditional coarse generation
            os.makedirs(args.output_dir, exist_ok=True)
            # Sample images
            num = args.num_samples
            shape = (args.batch_size, args.in_channels, args.image_size//(2**args.j), args.image_size//(2**args.j))
            all_imgs = []
            while len(all_imgs) < num:
                sample = diffusion.p_sample_loop(model, shape, device=model.device)
                # Unwhiten coarse
                mean, std = curvelet_datasets.curvelet_stats(args.j, args.data_dir)
                mean, std = torch.tensor(mean[:3]), torch.tensor(std[:3])
                sample = sample * std[None,:,None,None] + mean[None,:,None,None]
                sample = sample.clamp(-1, 1)
                for i in range(sample.size(0)):
                    img_arr = ((sample[i].cpu().numpy().transpose(1,2,0) + 1) * 127.5).astype(np.uint8)
                    img = Image.fromarray(img_arr)
                    img.save(os.path.join(args.output_dir, f"coarse_j{args.j}_{len(all_imgs)+i}.png"))
                all_imgs.extend([None]*sample.size(0))
                # (We don't actually store images in list; just counting)
                if len(all_imgs) >= num:
                    break
    else:
        # handle other tasks (standard, super_res, wavelet) similarly...
        pass

if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    main()
