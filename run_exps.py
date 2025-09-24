import os
import shlex
import subprocess
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--print", action="store_true", help="only print the commands instead of running them")
args = parser.parse_args()


class Exps:
    def __init__(self, exps):
        self.exps = exps  # List of exps, each exp being represented as tuple(names, cmds) which represent name and cmd parts to join.

    def __or__(self, other):
        return Exps(self.exps + other.exps)

    def __mul__(self, other):
        return Exps([(name1 + name2, cmd1 + cmd2) for name1, cmd1 in self.exps for name2, cmd2 in other.exps])

    def get_exps(self):
        """ Returns an iterator over list of pairs (name, cmd). """
        for names, cmds in self.exps:
            name = "-".join(filter(None, names))
            cmd = " ".join(filter(None, cmds))
            yield name, cmd

    def run(self):
        for i, (name, cmd) in enumerate(self.get_exps(), start=1):
            print(f"[{i:{len(str(len(self.exps)))}}/{len(self.exps)}] {name}:\t{cmd}")
            if not args.print:
                env = os.environ.copy()
                env.update(OPENAI_LOGDIR=f"logs/{name}")
                subprocess.run(shlex.split(cmd), check=True, env=env)


def exp(name="", cmd=""):
    return Exps([([name], [cmd])])


single_empty_exp = exp()  # A single exp with no arguments nor name
empty_exps = Exps([])  # An empty list of experiments


def ors(exps):
    res = empty_exps
    for exp in exps:
        res = res | exp
    return res


def base_exp(task, j=1, conditional=True, max_training_steps=500000, resume_step=None, num_channels=128, num_res_blocks=3, channel_mult=None, channel_mults=None, num_samples=128, sample=None, diffusion_steps=None, train_noise_schedule="quadratic", sample_noise_schedule="uniform", other_exps=single_empty_exp, dataset="cifar", data_suffix="", debug=False, batch_size=32):
    """ Builds a base experiment with all the standard flags.
    :param task: "standard" for single-scale generation, and "wavelet" for multiscale generations
    :param j: scale for wavelet generations of training, or maximum scale for sampling
    :param conditional: whether to predict low or high-frequencies for training, and whether to predict low-frequencies or use those of the test set for sampling
    :param sample: None for training a new model, or path to a model for sampling
    True means to infer the path to the model based on the given arguments
    :param diffusion_steps, noise_schedule: parameters for the discretization of diffusion
    :param other_exps: other experiments with additional exps
    """
    base_exps = []

    for other_name, other_cmd in other_exps.get_exps():
        # Name determination (except for special flags)
        def get_name(j, conditional, cmult):
            """ Builds the standard name for the given scale and sconditionality. """
            name = f"{dataset}-global" if task == "standard" else f"{dataset}-wavelet-{data_suffix}j{j}{'hi' if conditional else 'lo'}"
            name = f"{name}-{train_noise_schedule}"
            if cmult is not None:
                name = f"{name}-cmult{''.join(str(c) for c in cmult)}"
            if other_name != "":
                name = f"{name}-{other_name}"
            return name
        name = get_name(j, conditional, "".join(str(c) for c in channel_mult))

        def get_model_path(name, ema, step):
            """ Returns the model path (ema or not) for a given experiment name and step. """
            assert " "not in name
            return f"logs/{name}/{'ema_0.9999_' if ema else 'model'}{step:06d}.pt"

        # List of scales: [0], [j], [1..J], or [1..J, J]
        if task == "standard":
            scales = [0]
            conditionals = [conditional]
        elif sample is None:
            scales = [j]
            conditionals = [conditional]
        else:
            scales = list(range(1, j + 1)) + ([] if conditional else [j])
            conditionals = [True] * j + ([] if conditional else [False])

        # Flags determination.
        def f(flag):
            """ Replicate a flag if we use several models. """
            if sample is None:
                return flag  # Training, nothing to do
            else:
                # Sampling, replicate flag j[+1] times
                flags = [flag] * len(scales)
                return " ".join(str(f) for f in flags)

        if dataset == "cifar":
            if "periodic" in data_suffix:
                sizes = {0: 32, 1: 16, 2: 8, 3: 4, 4: 2}
            else:
                sizes = {0: 32, 1: 19, 2: 13, 3: 10, 4: 8}
        elif "128" in dataset:
            sizes = {0: 128, 1: 64, 2: 32}
        elif "256" in dataset:
            sizes = {0: 256, 1: 128, 2: 64, 3: 32}
        elif "512" in dataset:
            sizes = {0: 512, 1: 256, 2: 128, 3: 64, 4: 32}
        size = " ".join(str(sizes[i]) for i in scales)
        flags = f"--task {task} --large_size {size} --num_channels {f(num_channels)} --num_res_blocks {f(num_res_blocks)}"
        if channel_mult is not None:
            if channel_mults is None:
                long_channel_mults = [channel_mult] * len(scales)
            else:
                long_channel_mults = channel_mults + [channel_mult]
            short_channel_mults = [long_channel_mults[i][scales[i]:] for i in range(len(scales))]
            flags = f"{flags} --channel_mult {' '.join(','.join(str(c) for c in channel_mult) for channel_mult in short_channel_mults)}"
        flags = f"{flags} --noise_schedule {f(train_noise_schedule if sample is None else sample_noise_schedule)} --batch_size {batch_size}"

        if sample is None:
            # Train-specific flags
            flags = f"{flags} --lr 1e-4 --save_interval 10000 --log_interval 100"
            if max_training_steps is not None:
                flags = f"{flags} --max_training_steps {max_training_steps}"
            if resume_step is not None:
                flags = f"{flags} --resume_checkpoint {get_model_path(name, ema=False, step=resume_step)}"
        else:
            # Sample-specific flags, determine model path and create new sampling name.
            if not isinstance(sample, str):
                sample = " ".join(get_model_path(get_name(scale, cond, cmult), ema=True, step=max_training_steps) for scale, cond, cmult in zip(scales, conditionals, long_channel_mults))
            flags = f"{flags} --model_path {sample} --num_samples {num_samples} --diffusion_steps {f(diffusion_steps)} --wavelet {'haar' if 'haar' in data_suffix else 'db2' if 'db2' in data_suffix else 'db4'} --border_condition {'symmetric' if 'cifar' in dataset and 'periodic' not in data_suffix else 'periodization'} --final_size {sizes[0]}"
            name = f"{name}-sample{max_training_steps}-{sample_noise_schedule}{diffusion_steps}"

        # Data path and flags.
        data = f"datasets/{dataset}"
        if dataset == "cifar":
            data = f"{data}_{'train' if sample is None else 'test'}"
        if task == "wavelet":
            data = f"{data}_J4"
            c = " ".join(str(int(c)) for c in conditionals)
            flags = f"{flags} --j {j} --conditional {c}"
        if data_suffix != "":
            data = f"{data}_{data_suffix}"

        # Script with data-specific flags.
        if sample is None:
            script = f"image_train.py --data_dir {data}"
        else:
            script = f"image_sample.py"
            if task == "wavelet":
                script = f"{script} --wavelet_dir {data}"
                if conditional:
                    script = f"{script} --base_samples {data}"

        cmd = f"python scripts/{script} {flags} {other_cmd}"

        base_exps.append(exp(name=name, cmd=cmd))

    return ors(base_exps)


exps = empty_exps

# Comment or uncomment the lines.

# Training and sampling global model
exps |= base_exp(task="standard", dataset="celebA128", channel_mult="12244", max_training_steps=500000, batch_size=16)
# exps |= ors(base_exp(task="standard", dataset="celebA128", channel_mult="12244", max_training_steps=training_steps, sample=True, num_samples=30000, diffusion_steps=diffusion_steps) for diffusion_steps in [2 ** i for i in range(11)] for training_steps in [500000])

# Training and sampling wavelet models
exps |= ors(base_exp(task="wavelet", dataset="celebA128", data_suffix="periodic_haar", j=j, conditional=conditional, channel_mult="12244", batch_size=16 if j == 1 else 32, max_training_steps=500000) for j in range(1, 3) for conditional in [False, True] if j < 2 or conditional)
# exps |= ors(base_exp(task="wavelet", dataset="celebA128", data_suffix="periodic_haar", j=j, conditional=conditional, channel_mult="12244", sample=True, num_samples=30000, diffusion_steps=diffusion_steps) for j in [1] for conditional in [False] for diffusion_steps in [2 ** i for i in range(1, 11)] if j < 2 or conditional)

# Training and sampling unconditional model at resolution 32x32
exps |= ors(base_exp(task="wavelet", dataset="celebA128", data_suffix="periodic_haar", j=j, conditional=conditional, channel_mult="001222", batch_size=32, max_training_steps=500000) for j in [2] for conditional in [False])
# exps |= ors(base_exp(task="wavelet", dataset="celebA128", data_suffix="periodic_haar", j=j, conditional=conditional, channel_mult="001222", channel_mults=["12244", "12244"], batch_size=32, max_training_steps=500000, sample=True, num_samples=30000, diffusion_steps=diffusion_steps) for j in [2] for conditional in [False] for diffusion_steps in [2 ** i for i in range(1, 11)])

exps.run()
