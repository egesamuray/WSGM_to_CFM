# scripts/image_train.py
import os
import torch
from improved_diffusion import script_util
from improved_diffusion import wavelet_datasets, curvelet_datasets

def main():
    args = script_util.create_argparser().parse_args()
    # Derive device once and attach for convenience
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Update model channels for wavelet/curvelet tasks
    script_util.update_model_channels(args)

    # Create model & diffusion (works with args-namespace now)
    model, diffusion = script_util.create_model_and_diffusion(args)
    model.to(dev)
    # convenience: some code expects model.device
    setattr(model, "device", dev)

    # Prepare data loader
    if args.task == "wavelet":
        data = wavelet_datasets.load_data_wavelet(
            args.data_dir, args.batch_size, args.j,
            conditional=args.conditional, deterministic=False
        )
    elif args.task == "curvelet":
        data = curvelet_datasets.load_data_curvelet(
            args.data_dir, args.batch_size, args.j,
            conditional=args.conditional,
            angles_per_scale=[int(a) for a in args.angles_per_scale.split(',')] if args.angles_per_scale else None,
            deterministic=False
        )
    else:
        # Fallback (not used in your current runs)
        from improved_diffusion import image_datasets as datasets
        data = datasets.load_data(data_dir=args.data_dir, batch_size=args.batch_size, image_size=args.large_size, class_cond=False)

    # Training loop (toy)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step, batch in enumerate(data):
        model.train()
        optimizer.zero_grad()

        if args.task in ["wavelet", "curvelet"] and args.conditional:
            high_freq, cond = batch  # tuple from generator
            high_freq = high_freq.to(dev)
            cond = cond.to(dev)
            model_kwargs = {"low_res": cond}
            losses = diffusion.training_losses(model, high_freq, model_kwargs=model_kwargs)
        else:
            batch = batch.to(dev)
            losses = diffusion.training_losses(model, batch)

        loss = losses["loss"].mean()
        loss.backward()
        optimizer.step()

        if args.save_interval and step % args.save_interval == 0:
            save_dir = os.environ.get("OPENAI_LOGDIR", "")
            os.makedirs(save_dir or ".", exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir or ".", f"model_{step:06d}.pt"))

        if step >= args.iterations:
            break

if __name__ == "__main__":
    main()
