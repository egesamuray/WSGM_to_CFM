"""
Train a diffusion model on images.
"""

import argparse
import json

import torch.nn.functional as F

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.wavelet_datasets import load_data_wavelet
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    with open(f"{logger.get_dir()}/args.json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        task=args.task, **args_to_dict(args, model_and_diffusion_defaults(task=args.task).keys()),
    )
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    if args.task == "standard":
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.large_size,
            class_cond=args.class_cond,
        )
    elif args.task == "super_res":
        data = load_superres_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            large_size=args.large_size,
            small_size=args.small_size,
            class_cond=args.class_cond,
        )
    elif args.task == "wavelet":
        data = load_data_wavelet(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            j=args.j,
            conditional=args.conditional,
        )
    else:
        assert False

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=diffusion.schedule,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        max_training_steps=args.max_training_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        task="standard",
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        max_training_steps=500000,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["conditional"] = F.interpolate(large_batch, small_size, mode="area")
        yield large_batch, model_kwargs


if __name__ == "__main__":
    main()
