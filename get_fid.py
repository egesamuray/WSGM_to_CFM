import os
import shlex
import subprocess
import argparse


num_discretizations = 10


parser = argparse.ArgumentParser()
parser.add_argument("--print", action="store_true", help="only print the commands instead of running them")
args = parser.parse_args()

for final_scale in range(0, 3):
    cmd = f"python scripts/fid.py"
    cmd = f"{cmd} --dump logs/fid_celebA128_j{final_scale}"
    paths = []
    if final_scale == 0:
        paths.append("datasets/celebA128")
    else:
        cmd = f"{cmd} --j {final_scale}"
        paths.append("datasets/celebA128_J4_periodic_haar")

    for start_scale in range(final_scale, 3):
        for num_steps in [2 ** i for i in range(1, num_discretizations + 1)]:
            for schedule in ["quadratic", "uniform"]:
                cmults = ["12244"]
                if start_scale == 2:
                    cmults = ["001222"]
                for cmult in cmults:
                    path = "logs/celebA128"
                    if start_scale == 0:
                        path = f"{path}-global"
                    else:
                        path = f"{path}-wavelet-periodic_haarj{start_scale}lo"
                    path = f"{path}-quadratic-cmult{cmult}"
                    path = f"{path}-sample500000-{schedule}{num_steps}"
                    size = 128 // (2 ** final_scale)
                    path = f"{path}/samples_30000x{size}x{size}x3.npz"
                    if not os.path.exists(path):
                        print(f"Skipped path {path}")
                    else:
                        paths.append(path)

    print(*paths, sep="\n")
    if not args.print:
        cmd = f"{cmd} {' '.join(paths)}"
        subprocess.run(shlex.split(cmd), check=True)
