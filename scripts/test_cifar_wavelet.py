# scripts/test_cifar_wavelet.py
import os, json, torch
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict, add_dict_to_argparser
)
from improved_diffusion.wavelet_datasets import load_data_wavelet

def main():
    # --- args (tiny config for smoke test) ---
    defaults = dict(
        task="wavelet", data_dir="datasets/cifar_train_J2_periodic_haar",
        j=1, conditional=True, batch_size=16, lr=1e-4,
        max_training_steps=50, log_interval=10, save_interval=1000,
        num_channels=64, num_res_blocks=1, channel_mult="122",  # smaller model
        diffusion_steps=256, noise_schedule="quadratic",
    )
    defaults.update(model_and_diffusion_defaults(task="wavelet"))
    import argparse
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args([])  # run with defaults

    # --- setup ---
    dist_util.setup_dist()
    logger.configure()
    with open(f"{logger.get_dir()}/args.json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # --- model & diffusion ---
    model, diffusion = create_model_and_diffusion(
        task=args.task, **args_to_dict(args, model_and_diffusion_defaults(task=args.task).keys())
    )
    model.to(dist_util.dev())

    # --- data ---
    data = load_data_wavelet(
        data_dir=args.data_dir, batch_size=args.batch_size, j=args.j, conditional=args.conditional
    )

    # --- train a few steps ---
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step, (x, kw) in zip(range(args.max_training_steps), data):
        x = x.to(dist_util.dev())
        kw = {k: v.to(dist_util.dev()) for k, v in kw.items()}
        model.train(); opt.zero_grad()
        losses = diffusion.training_losses(model, x, model_kwargs=kw)
        loss = losses["loss"].mean()
        loss.backward(); opt.step()
        if step % args.log_interval == 0:
            logger.logkv_mean("loss", loss.item()); logger.dumpkvs()
    print("Smoke test complete.")

if __name__ == "__main__":
    main()
