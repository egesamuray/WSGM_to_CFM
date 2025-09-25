# run_exps.py
import os
import subprocess
import glob
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["wavelet","curvelet"], default="curvelet")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of training images")
    parser.add_argument("--j", type=int, required=True, help="Number of scales (for wavelet/curvelet) or scale to train.")
    parser.add_argument("--angles_per_scale", type=str, default=None, help="Angles per scale (comma-separated) for curvelet.")
    parser.add_argument("--final_size", type=int, default=None, help="Final image size (px). If not given, will use dataset image size.")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of images to sample at the end.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # If final_size not provided, try to infer from data (e.g., by reading first image)
    final_size = args.final_size
    if final_size is None:
        import glob
        from PIL import Image
        img_files = glob.glob(os.path.join(args.data_dir, "*"))
        if img_files:
            with Image.open(img_files[0]) as img:
                final_size = max(img.size)
        else:
            raise ValueError("Could not infer final image size; please specify --final_size")

    # Derive default angles if not provided for curvelet
    if args.task == "curvelet" and args.angles_per_scale is None:
        # Example default as described in plan
        angles = []
        ang = 8
        for i in range(args.j):
            angles.append(min(ang, 32))
            if ang < 32:
                ang *= 2
        args.angles_per_scale = ",".join(map(str, angles))

    # Directory to store results
    exp_name = f"{args.task}_J{args.j}"
    results_dir = os.path.join("results", exp_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to {results_dir}")
    # Precompute stats (curvelet_stats will save to file)
    if args.task == "curvelet":
        for scale in range(1, args.j+1):
            print(f"Computing stats for scale {scale}...")
            # Compute stats by calling curvelet_stats (which caches to npz)
            from improved_diffusion import curvelet_datasets
            curvelet_datasets.curvelet_stats(scale, args.data_dir, angles_per_scale=[int(a) for a in args.angles_per_scale.split(',')])

    # Train models
    model_paths = {}
    if args.task == "curvelet":
        # Train coarsest coarse model
        log_dir = os.path.join(results_dir, f"curvelet_coarse_J{args.j}")
        os.makedirs(log_dir, exist_ok=True)
        print(f"Training coarse model (scale {args.j})...")
        subprocess.run([
            "python", "scripts/image_train.py",
            "--task", "curvelet", "--j", str(args.j), "--conditional", "False",
            "--data_dir", args.data_dir, "--lr", str(args.lr),
            "--batch_size", str(args.batch_size), "--angles_per_scale", args.angles_per_scale
        ], check=True)
        # Find last checkpoint
        ckpts = sorted(glob.glob(os.path.join(log_dir, "*.pt")))
        model_paths[f"coarse_J{args.j}"] = ckpts[-1] if ckpts else None
        # Train conditional models for each scale
        for scale in range(args.j, 0, -1):
            log_dir = os.path.join(results_dir, f"curvelet_detail_j{scale}")
            os.makedirs(log_dir, exist_ok=True)
            print(f"Training detail model for scale {scale}...")
            subprocess.run([
                "python", "scripts/image_train.py",
                "--task", "curvelet", "--j", str(scale), "--conditional", "True",
                "--data_dir", args.data_dir, "--lr", str(args.lr),
                "--batch_size", str(args.batch_size), "--angles_per_scale", args.angles_per_scale
            ], check=True)
            ckpts = sorted(glob.glob(os.path.join(log_dir, "*.pt")))
            model_paths[f"detail_j{scale}"] = ckpts[-1] if ckpts else None

    elif args.task == "wavelet":
        # Similar block for wavelet (omitted for brevity)
        pass

    # Sampling
    samples_dir = os.path.join(results_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    if args.task == "curvelet":
        # Sample coarse first
        coarse_out = os.path.join(samples_dir, f"coarse_J{args.j}")
        os.makedirs(coarse_out, exist_ok=True)
        print("Sampling coarsest images...")
        subprocess.run([
            "python", "scripts/image_sample.py",
            "--task", "curvelet", "--j", str(args.j), "--conditional", "False",
            "--data_dir", args.data_dir,
            "--model_path", model_paths.get(f"coarse_J{args.j}", ""),
            "--output_dir", coarse_out,
            "--num_samples", str(args.num_samples),
            "--batch_size", str(min(args.batch_size, args.num_samples)),
            "--image_size", str(final_size)
        ], check=True)
        # Iteratively sample each detail scale
        current_dir = coarse_out
        for scale in range(args.j, 0, -1):
            detail_out = os.path.join(samples_dir, f"detail_j{scale}")
            os.makedirs(detail_out, exist_ok=True)
            print(f"Sampling detail scale {scale}...")
            subprocess.run([
                "python", "scripts/image_sample.py",
                "--task", "curvelet", "--j", str(scale), "--conditional", "True",
                "--data_dir", args.data_dir,
                "--model_path", model_paths.get(f"detail_j{scale}", ""),
                "--cond_dir", current_dir, "--output_dir", detail_out,
                "--angles_per_scale", args.angles_per_scale
            ], check=True)
            current_dir = detail_out
        print(f"Final images saved in {current_dir}")
    elif args.task == "wavelet":
        # Similar multi-stage sampling for wavelet (omitted)
        pass

    print("Done. You can compute FID on the final images folder if desired.")

if __name__ == "__main__":
    main()

