# scripts/image_train.py
import os
import torch
from improved_diffusion import script_util
from improved_diffusion import wavelet_datasets, curvelet_datasets

def main():
    args = script_util.create_argparser().parse_args()
    # Update model channels for wavelet/curvelet tasks
    script_util.update_model_channels(args)
    # Create model & diffusion
    model, diffusion = script_util.create_model_and_diffusion(args)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
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
            angles_per_scale = [int(a) for a in args.angles_per_scale.split(',')] if args.angles_per_scale else None,
            deterministic=False
        )
    else:
        # Fallback to standard image loading (not shown)
        from improved_diffusion import datasets
        data = datasets.load_data(data_dir=args.data_dir, batch_size=args.batch_size)
    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step, batch in enumerate(data):
        model.train()
        optimizer.zero_grad()
        if args.task in ["wavelet", "curvelet"] and args.conditional:
            high_freq, cond = batch  # tuple from generator
            high_freq = high_freq.to(model.device)
            cond = cond.to(model.device)
            # Diffusion training step: predict noise for high_freq, condition on cond
            # Assuming model call signature allows passing cond as extra input or model itself merges conditioning.
            model_kwargs = {"low_res": cond} if args.task in ["super_res", "curvelet", "wavelet"] else {}
            losses = diffusion.training_losses(model, high_freq, model_kwargs=model_kwargs)
        else:
            # Unconditional case
            batch = batch.to(model.device)
            losses = diffusion.training_losses(model, batch)
        loss = losses["loss"].mean()
        loss.backward()
        optimizer.step()
        # Save or log periodically
        if step % args.save_interval == 0:
            save_path = os.path.join(os.environ.get("OPENAI_LOGDIR", ""), f"model_{step:06d}.pt")
            torch.save(model.state_dict(), save_path)
        if step >= args.iterations:
            break

if __name__ == "__main__":
    main()
