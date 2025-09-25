# improved_diffusion/script_util.py
import argparse

from . import gaussian_diffusion as gd
from .resample import ScheduleSampler
from .unet import SuperResModel, UNetModel, ConditionalModel

NUM_CLASSES = 1000

"""
In this file, the argument `task` represents the task performed by the model:
- "standard": unconditional denoising
- "super_res": super-resolution (denoising conditioned on low-resolution input)
- "wavelet": denoising of high-frequencies conditioned on low-frequencies
- "curvelet": denoising of curvelet high-frequency wedges conditioned on the coarse image
"""


# ------------------------------- Defaults ------------------------------------


def model_and_diffusion_defaults(task: str = "standard"):
    """
    Defaults for image training. For `super_res`, we keep a larger default `large_size`.
    For other tasks, defaults mirror the original repo.
    """
    large_size = 256 if task == "super_res" else 64
    small_size = 64 if task == "super_res" else 64

    return dict(
        # Task-specific spatial defaults
        large_size=large_size,
        small_size=small_size,

        # Multiscale controls
        j=0,                      # Scale for wavelet/curvelet (1 = finest, coarser as j increases). Kept 0 for BC.
        conditional=True,         # For wavelet/curvelet: True => predict high-freq given coarse; False => unconditional coarse

        # (New) curvelet wedges per scale (string "8,16,32,32" from coarsest->finest). None => determine elsewhere.
        angles_per_scale=None,

        # UNet / attention hyperparameters
        channel_mult=None,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,

        # Diffusion / loss config
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=-1,
        final_time=5.0,
        noise_schedule="quadratic",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,

        # Training utils
        use_checkpoint=False,
        use_scale_shift_norm=True,
        learn_potential=False,

        # I/O channel hints (will be overridden by update_model_channels/create_model)
        in_channels=3,
        conditioning_channels=0,
    )


# --------------------------- Model + Diffusion -------------------------------


def create_model_and_diffusion(
    task,
    large_size,
    small_size,
    j,
    conditional,
    class_cond,
    learn_sigma,
    sigma_small,
    channel_mult,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    final_time,
    noise_schedule,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    learn_potential,
    angles_per_scale=None,  # NEW: pass through to create_model for curvelet
):
    model = create_model(
        task=task,
        large_size=large_size,
        small_size=small_size,
        j=j,
        conditional=conditional,
        channel_mult=channel_mult,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        learn_potential=learn_potential,
        angles_per_scale=angles_per_scale,  # NEW
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        final_time=final_time,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
    )
    return model, diffusion


def _parse_angles(angles_per_scale):
    """
    Accepts None | list[int] | str like "8,16,32,32".
    Returns list[int] (coarsest -> finest) or None.
    """
    if angles_per_scale is None:
        return None
    if isinstance(angles_per_scale, (list, tuple)):
        return [int(x) for x in angles_per_scale]
    if isinstance(angles_per_scale, str):
        s = angles_per_scale.strip()
        if not s:
            return None
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    raise TypeError("angles_per_scale must be None, list[int], or comma-separated string")


def _wedges_at_scale(j: int, angles_per_scale):
    """
    Given scale index j (1=finest), and angles_per_scale (coarsest->finest),
    return W_j = angles_per_scale[-j]. If unavailable, fall back to 1.
    """
    angles = _parse_angles(angles_per_scale)
    if not angles or j is None or j <= 0:
        return 1  # conservative fallback; pass explicit --angles_per_scale to set correctly
    idx = -j
    idx = max(-len(angles), idx)  # clamp
    return int(angles[idx])


def create_model(
    task,
    large_size,
    small_size,
    j,
    conditional,
    channel_mult,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    learn_potential,
    angles_per_scale=None,  # NEW
):
    _ = small_size  # hack to prevent unused variable

    if channel_mult is None:
        if large_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif large_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif large_size == 32:
            channel_mult = (1, 2, 2, 2)
        else:
            raise ValueError(f"unsupported large size: {large_size}")
    else:
        # Convert "1,2,3" -> (1,2,3)
        channel_mult = tuple(map(int, channel_mult.split(",")))

    attention_ds = [large_size // int(res) for res in attention_resolutions.split(",")]

    # Select model class by task
    if task == "standard":
        model_cls = UNetModel
    elif task == "super_res":
        model_cls = SuperResModel
    elif task == "wavelet":
        model_cls = ConditionalModel if conditional else UNetModel
    elif task == "curvelet":
        model_cls = ConditionalModel if conditional else UNetModel
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Base kwargs
    kwargs = dict(
        in_channels=3,  # will be overridden below for wavelet/curvelet conditional
        model_channels=num_channels,
        in_space=large_size,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        learn_potential=learn_potential,
    )

    # Wavelet conditional: 3 subbands × 3 colors = 9 channels, conditioned on 3-channel coarse
    if task == "wavelet" and conditional:
        kwargs.update(in_channels=9, conditioning_channels=3)

    # Curvelet conditional: 3 * W_j channels, conditioned on 3-channel coarse
    if task == "curvelet":
        if conditional:
            W_j = _wedges_at_scale(j if j is not None else 1, angles_per_scale)
            kwargs.update(in_channels=3 * W_j, conditioning_channels=3)
        else:
            kwargs.update(in_channels=3, conditioning_channels=0)

    # Output channels mirror input channels unless learning sigma
    kwargs.update(out_channels=kwargs["in_channels"] * (2 if learn_sigma else 1))

    return model_cls(**kwargs)


def create_gaussian_diffusion(
    *,
    steps=-1,
    final_time=5.0,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
):
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    return gd.GaussianDiffusion(
        schedule=ScheduleSampler(final_time=final_time, schedule=noise_schedule),
        num_diffusion_steps=steps,
        model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
        model_var_type=(
            (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


# ------------------------------- Argparse ------------------------------------


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys, j=0):
    """
    Returns a dictionary of key=value from arguments and an iterable of keys.
    Extracts the j-th item from list-valued arguments (nargs='+').
    """
    def get_arg(k):
        arg = getattr(args, k)
        if isinstance(arg, list):
            arg = arg[j]
        return arg
    return {k: get_arg(k) for k in keys}


def str2bool(v):
    """
    Parse boolean values from strings.
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    lv = v.lower()
    if lv in ("yes", "true", "t", "y", "1"):
        return True
    if lv in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def create_argparser():
    """
    Convenience parser used by scripts. We first parse --task to obtain task-aware defaults,
    then add the rest of the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["standard", "super_res", "wavelet", "curvelet"],
        default="standard",
        help="Task type.",
    )

    # Two-phase parsing: read --task to build task-aware defaults
    known_args, _ = parser.parse_known_args()
    defaults = model_and_diffusion_defaults(task=known_args.task)
    add_dict_to_argparser(parser, defaults)

    # Common script-level args (not consumed by create_model_and_diffusion)
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=100000, help="Training iterations")
    parser.add_argument("--save_interval", type=int, default=10000, help="Steps between checkpoints")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a .pt checkpoint (for sampling)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (for sampling)")
    parser.add_argument("--cond_dir", type=str, default=None, help="Directory with conditioning images (sampling)")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate (sampling)")
    parser.add_argument("--image_size", type=int, default=None, help="Full-resolution image size (sampling)")

    # Optional help text for curvelet wedges (already added via defaults)
    parser.add_argument(
        "--angles_per_scale",
        type=str,
        default=defaults.get("angles_per_scale", None),
        help="Comma-separated wedges per scale (coarsest->finest), e.g., '8,16,32,32'.",
    )
    return parser


# ------------------------ Task-aware channel updates -------------------------


def update_model_channels(args):
    """
    Set args.in_channels / args.conditioning_channels / args.out_channels
    based on the selected task. For curvelet, uses angles_per_scale to derive W_j
    (number of wedges at scale j). If angles are unknown, uses a safe placeholder.
    """
    # Defaults
    args.in_channels = getattr(args, "in_channels", 3)
    args.conditioning_channels = getattr(args, "conditioning_channels", 0)
    learn_sigma = getattr(args, "learn_sigma", False)

    if args.task == "wavelet":
        if args.conditional:
            args.in_channels = 9  # 3 subbands × 3 colors
            args.conditioning_channels = 3
        else:
            args.in_channels = 3
            args.conditioning_channels = 0

    elif args.task == "curvelet":
        if args.conditional:
            # number of wedges at scale j:
            angles = _parse_angles(getattr(args, "angles_per_scale", None))
            if angles:
                # angles are listed coarsest->finest; j=1 is finest
                W_j = angles[-1] if len(angles) == 1 else angles[-max(1, int(args.j))]
            else:
                # Placeholder until stats/dataset define it explicitly
                W_j = 1
            args.in_channels = 3 * int(W_j)
            args.conditioning_channels = 3
        else:
            args.in_channels = 3
            args.conditioning_channels = 0

    # Mirror input channels in output unless learning sigma
    args.out_channels = int(args.in_channels) * (2 if learn_sigma else 1)
