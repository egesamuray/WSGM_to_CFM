"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.wavelet_datasets import load_data_wavelet, wavelet_to_image, wavelet_stats
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    str2bool,
)


def main():
    args = parse_args()

    """ There are several possibilities:
    - task = standard: no base_samples
    - task = super_res: base_samples
    - task = wavelet: --j defines the maximum scale J, we expect J conditional model plus either base_sample or
    an unconditional J+1-th model to predict low frequencies at scale J.
    """
    if args.task in ["standard", "super_res"]:
        num_models = 1
    elif args.task == "wavelet":
        num_models = args.j + (args.base_samples == "")
    else:
        assert False

    dist_util.setup_dist()
    logger.configure()

    logger.log(f"creating {num_models} models and diffusions...")

    models, diffusions = [], []
    for j in range(num_models):
        model, diffusion = create_model_and_diffusion(
            task=args.task, **args_to_dict(args, model_and_diffusion_defaults(task=args.task).keys(), j=j)
        )
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path[j], map_location="cpu")
        )
        model.to(dist_util.dev())
        model.eval()

        models.append(model)
        diffusions.append(diffusion)

    if args.task == "super_res":
        logger.log("loading data...")
        data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)
    elif args.task == "wavelet":
        if args.base_samples != "":
            # There is no model for unconditionally generating low-frequencies, need base samples.
            data = load_data_for_worker_wavelet(
                base_samples=args.base_samples,
                batch_size=args.batch_size,
                j=args.j,
            )
        means, stds = [], []
        for j in range(1, args.j + 1):
            mean, std = wavelet_stats(j, args.wavelet_dir)
            means.append(mean.to(dist_util.dev()))
            stds.append(std.to(dist_util.dev()))

    logger.log("sampling...")
    all_images = [[] for _ in range(num_models)]  # j -> list of images
    all_labels = []
    while len(all_images[0]) * args.batch_size < args.num_samples:

        low_freq = None  # low frequencies (whitened) (B, 3, N_j, N_j) which we use to generate the next scale.
        generations_j = []  # list of generated images at each scale (B, 3, N_j, N_j) in [0, 255].

        for j in range(num_models - 1, -1, -1):  # j ranges from J [- 1] to 0
            # Prepare model_kwargs for conditioning.
            if j == num_models - 1:
                # First model, might need a base sample.
                model_kwargs = next(data) if args.base_samples != "" else {}
                model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
            else:
                # Later models, use previous generation.
                model_kwargs = dict(conditional=low_freq)
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes

            # We predict high-frequencies if we are in wavelet mode and we are not using the (optional) J+1-th model.
            predict_high_freq = args.task == "wavelet" and j < args.j

            # Get a sample from the model.
            sample_fn = (
                diffusions[j].p_sample_loop if not args.use_ddim else diffusions[j].ddim_sample_loop
            )
            out_channels = 9 if predict_high_freq else 3
            shape = (args.batch_size, out_channels, args.large_size[j], args.large_size[j])
            # logger.log(f"Sampling shape {shape} conditioning shape {model_kwargs['conditional'].shape if 'conditional' in model_kwargs else None}")
            sample = sample_fn(
                models[j],
                shape,
                clip_denoised=args.clip_denoised and not predict_high_freq and j == 0,
                model_kwargs=model_kwargs,
            )  # Can be: normal image, low-freq J, or high-freq

            # Prepare for next model.
            if args.task == "wavelet":
                if predict_high_freq:
                    # Convert high-frequencies to next low-frequencies.
                    white_wavelet_coeffs = th.cat((sample, model_kwargs["conditional"]), dim=1)  # (B, 12, h, w)
                    wavelet_coeffs = white_wavelet_coeffs * stds[j][:, None, None] + means[j][:, None, None]
                    next_low_freq = th.from_numpy(wavelet_to_image(
                        wavelet_coeffs.cpu().numpy(), wavelet=args.wavelet, border_condition=args.border_condition,
                        output_size=args.final_size if j == 0 else args.large_size[j - 1],
                    )).to(dist_util.dev())
                    if j > 0:
                        # Prepare low-freq for next model.
                        low_freq = (next_low_freq - means[j - 1][-3:, None, None]) / stds[j - 1][-3:, None, None]
                        # Also add to generations, normalizing to [0, 255] in a quick and dirty way.
                        generations_j.append((low_freq - low_freq.min()) * 255 / (low_freq.max() - low_freq.min()))
                    else:
                        # Final loop, add the final image with values in [0, 255] to the generations.
                        generations_j.append(next_low_freq)
                else:
                    # sample is already white low_frequencies for next model.
                    low_freq = sample
                    # Also add to generations, normalizing to [0, 255] in a quick and dirty way.
                    generations_j.append((low_freq - low_freq.min()) * 255 / (low_freq.max() - low_freq.min()))
            else:
                # Final (and unique loop), add the final image with values in [0, 255] to the (empty) generations.
                generations_j.append((sample + 1) * 127.5)

        for i, sample in enumerate(generations_j):
            # Convert generations from  (B, C, H, W) [0, 255] float to NHWC uint8.
            sample = sample.clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1).contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images[i].extend([sample.cpu().numpy() for sample in gathered_samples])
            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images[i]) * args.batch_size} samples")

    for all_images_j in all_images:
        arr = np.concatenate(all_images_j, axis=0)
        arr = arr[: args.num_samples]
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if args.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if class_cond:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(conditional=batch)
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


def load_data_for_worker_wavelet(base_samples, batch_size, j):
    """ Loads low frequencies at scale j for conditional generation. """
    data = load_data_wavelet(
            data_dir=base_samples,
            batch_size=batch_size,
            j=j,
            conditional=True,
        )
    for x, model_kwargs in data:
        yield model_kwargs


def parse_args():
    defaults = dict(
        task="standard",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",  # numpy array for "super_res", folder of wavelet coefficients for "wavelet" (conditional generations)
        wavelet_dir="",  # same as base_samples but always needed for mean/std of dataset...
        final_size=32,
        wavelet="db4",
        border_condition="symmetric",
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()

    unique_args = ["task", "small_size", "j", "num_samples", "batch_size", "base_samples", "class_cond",
                   "clip_denoised", "final_size", "wavelet_dir", "wavelet", "border_condition"]

    for k, v in defaults.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        nargs = None if k in unique_args else '+'
        parser.add_argument(f"--{k}", default=v, type=v_type, nargs=nargs)

    return parser.parse_args()


if __name__ == "__main__":
    main()
