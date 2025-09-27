# run_exps.py
import argparse
import os
import sys
import glob
import subprocess
import numpy as np

from improved_diffusion import curvelet_datasets


def _env_with_repo_on_path() -> dict:
    env = os.environ.copy()
    repo_root = os.path.abspath(os.path.dirname(__file__))
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_root if not old else f"{repo_root}{os.pathsep}{old}"
    return env


def _latest_pt(d):
    cands = [p for p in glob.glob(os.path.join(d, "*.pt")) if os.path.isfile(p)]
    if not cands:
        return None
    ema = [p for p in cands if "ema" in os.path.basename(p).lower()]
    if ema:
        ema.sort(key=os.path.getmtime, reverse=True)
        return ema[0]
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["standard", "super_res", "wavelet", "curvelet"], required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--j", type=int, default=1, help="Scale index (1=finest).")
    p.add_argument("--angles_per_scale", type=str, default=None, help="coarsest->finest, e.g. '8' or '8,16,32'")
    p.add_argument("--final_size", type=int, default=64, help="full image size (H=W)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_samples", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    assert args.task == "curvelet", "This runner currently wires the curvelet task only."

    results_dir = os.path.join("results", f"curvelet_J{args.j}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to {results_dir}")

    # 0) stats (CPU, safe)
    print(f"Computing stats for scale {args.j}...")
    mean, std = curvelet_datasets.curvelet_stats(
        j=args.j,
        dir_name=args.data_dir,
        angles_per_scale=args.angles_per_scale,
        image_size=args.final_size,
        limit=None,
        device="cpu",
    )
    np.savez(os.path.join(results_dir, f"curvelet_stats_j{args.j}.npz"),
             mean=mean.numpy(), std=std.numpy())

    env = _env_with_repo_on_path()
    repo_root = os.path.abspath(os.path.dirname(__file__))

    # 1) train coarse
    coarse_logdir = os.path.join(results_dir, "coarse")
    os.makedirs(coarse_logdir, exist_ok=True)
    env_coarse = dict(env, OPENAI_LOGDIR=coarse_logdir)
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
    subprocess.run(cmd_coarse, check=True, cwd=repo_root, env=env_coarse)
    coarse_ckpt = _latest_pt(coarse_logdir)
    if not coarse_ckpt:
        raise FileNotFoundError(f"No checkpoints found under {coarse_logdir}")

    # 2) train conditional
    cond_logdir = os.path.join(results_dir, "cond")
    os.makedirs(cond_logdir, exist_ok=True)
    env_cond = dict(env, OPENAI_LOGDIR=cond_logdir)
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
    subprocess.run(cmd_cond, check=True, cwd=repo_root, env=env_cond)
    cond_ckpt = _latest_pt(cond_logdir)
    if not cond_ckpt:
        raise FileNotFoundError(f"No checkpoints found under {cond_logdir}")

    # 3) sample final images end-to-end with both models
    print("Sampling...")
    samples_dir = os.path.join(results_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    cmd_sample = [
        sys.executable, "scripts/image_sample.py",
        "--task", "curvelet",
        "--j", str(args.j),
        "--angles_per_scale", args.angles_per_scale or "",
        "--data_dir", args.data_dir,
        "--num_samples", str(args.num_samples),
        "--large_size", str(args.final_size),
        "--small_size", str(args.final_size),
        "--coarse_model_path", coarse_ckpt,
        "--cond_model_path", cond_ckpt,
        "--output_dir", samples_dir,
    ]
    subprocess.run(cmd_sample, check=True, cwd=repo_root, env=env)

    print(f"Done. See {samples_dir}")


if __name__ == "__main__":
    main()

