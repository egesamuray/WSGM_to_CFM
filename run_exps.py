# run_exps.py
import argparse
import os
import sys
import subprocess
import numpy as np

from improved_diffusion import curvelet_datasets


def _env_with_repo_on_path() -> dict:
    """Ensure child Python processes can import 'improved_diffusion'."""
    env = os.environ.copy()
    repo_root = os.path.abspath(os.path.dirname(__file__))
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (repo_root if not old else f"{repo_root}{os.pathsep}{old}")
    return env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["standard", "super_res", "wavelet", "curvelet"], required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--j", type=int, default=1, help="Scale index (1=finest).")
    p.add_argument("--angles_per_scale", type=str, default=None, help="coarsest->finest, e.g. '8,16,16'")
    p.add_argument("--final_size", type=int, default=64, help="image size (H=W)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_samples", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = os.path.join("results", f"{args.task}_J{args.j}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to {results_dir}")

    # Precompute stats for curvelet (fast stream over images)
    if args.task == "curvelet":
        print(f"Computing stats for scale {args.j}...")
        mean, std = curvelet_datasets.curvelet_stats(
            j=args.j,
            dir_name=args.data_dir,
            angles_per_scale=args.angles_per_scale,
            image_size=args.final_size,
            limit=None,  # full set
        )
        np.savez(os.path.join(results_dir, f"curvelet_stats_j{args.j}.npz"),
                 mean=mean.numpy(), std=std.numpy())

        env = _env_with_repo_on_path()
        repo_root = os.path.abspath(os.path.dirname(__file__))

        # 1) Train coarse (unconditional) at this scale
        print(f"Training coarse model (scale {args.j})...")
        cmd_coarse = [
            sys.executable, "scripts/image_train.py",
            "--task", "curvelet",
            "--j", str(args.j),
            "--conditional", "False",
            "--data_dir", args.data_dir,
            "--lr", str(args.lr),
            "--batch_size", str(args.batch_size),
            "--angles_per_scale", args.angles_per_scale or "",
            "--large_size", str(args.final_size),
            "--small_size", str(args.final_size),
        ]
        subprocess.run(cmd_coarse, check=True, cwd=repo_root, env=env)

        # 2) Train wedges (conditional) at this scale
        print(f"Training conditional (wedges) model (scale {args.j})...")
        cmd_cond = [
            sys.executable, "scripts/image_train.py",
            "--task", "curvelet",
            "--j", str(args.j),
            "--conditional", "True",
            "--data_dir", args.data_dir,
            "--lr", str(args.lr),
            "--batch_size", str(args.batch_size),
            "--angles_per_scale", args.angles_per_scale or "",
            "--large_size", str(args.final_size),
            "--small_size", str(args.final_size),
        ]
        subprocess.run(cmd_cond, check=True, cwd=repo_root, env=env)

        # 3) Sample
        print("Sampling...")
        cmd_sample = [
            sys.executable, "scripts/image_sample.py",
            "--task", "curvelet",
            "--j", str(args.j),
            "--angles_per_scale", args.angles_per_scale or "",
            "--data_dir", args.data_dir,
            "--num_samples", str(args.num_samples),
            "--large_size", str(args.final_size),
            "--small_size", str(args.final_size),
        ]
        subprocess.run(cmd_sample, check=True, cwd=repo_root, env=env)

    else:
        raise NotImplementedError("This runner currently wires the curvelet task only.")


if __name__ == "__main__":
    main()

