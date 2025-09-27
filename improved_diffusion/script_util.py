# improved_diffusion/script_util.py
import argparse

from . import gaussian_diffusion as gd
from .resample import ScheduleSampler
from .unet import SuperResModel, UNetModel, ConditionalModel

NUM_CLASSES = 1000

"""
Tasks:
- "standard": unconditional denoising
- "super_res": upsampling/conditional denoising
- "wavelet": denoise wavelet high-freqs conditioned on low-freqs
- "curvelet": denoise curvelet wedges conditioned on coarse
"""


# ------------------------------- Defaults ------------------------------------
def model_and_diffusion_defaults(task: str = "standard"):
    """
    Defaults for image training. For `super_res`, keep a larger default `large_size`.
    """
    large_size = 256 if task == "super_res" else 64
    small_size = 64 if task == "super_res" else 64

    return dict(
        # Spatial
        large_size=large_size,
        small_size=small_size,

        # Multiscale
        j=0,                      # (1 = finest) used for curvelet/wavelet; kept 0 for BC with old scripts
        conditional=True,         # curvelet/wavelet: HF given coarse vs unconditional coarse
        angles_per_scale=None,    # "coarsest->finest", e.g. "8,16,16"

        # UNet / attention
        channel_mult=None,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,

        # Diffusion / losses
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
    angles_per_scale=None,  # NEW: plumb through to model
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
        angles_per_scale=angles_per_scale,
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
    Accepts None | list[int] | str like "8,16,32".
    Returns list[int] (coarsest -> finest) or None.
    """
    if angles_per_scale is None:
        return None
    if isinstance(angles_per_scale, (list, tuple)):
        return [int(x) for x in angles_per_scale]
    s = str(angles_per_scale).strip()
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _wedges_at_scale(j: int, angles_per_scale):
    """
    Given scale index j (1=finest), and angles_per_scale (coarsest->finest),
    return W_j = angles_per_scale[-j]. If unavailable, fall back to 1.
    """
    angles = _parse_angles(angles_per_scale)
    if not angles or j is None or j <= 0:
        return 1
    idx = max(-len(angles), -int(j))
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
    angles_per_scale=None,
):
    _ = small_size

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
        channel_mult = tuple(map(int, channel_mult.split(",")))

    attention_ds = [large_size // int(res) for res in attention_resolutions.split(",")]

    # pick model class
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

    kwargs = dict(
        in_channels=3,  # overridden below when needed
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

    # wavelet conditional: 9 in, cond=3
    if task == "wavelet" and conditional:
        kwargs.update(in_channels=9)
        kwargs["conditioning_channels"] = 3  # only ConditionalModel accepts it

    # curvelet: 3*W_j in, cond=3 when conditional; 3 in when unconditional
    if task == "curvelet":
        if conditional:
            W_j = _wedges_at_scale(j if j else 1, angles_per_scale)
            kwargs.update(in_channels=3 * W_j)
            kwargs["conditioning_channels"] = 3  # ConditionalModel only
        else:
            kwargs.update(in_channels=3)

    # Outputs mirror inputs unless we learn sigma
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
    """
    Construct GaussianDiffusion with repo's expected signature and normalize
    a couple of small differences between training and sampling call sites.
    """
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    # >>> IMPORTANT: this repo's GaussianDiffusion expects 'num_diffusion_steps'
    diff = gd.GaussianDiffusion(
        schedule=ScheduleSampler(final_time=final_time, schedule=noise_schedule),
        num_diffusion_steps=steps,  # <-- correct kwarg for this codebase
        model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
        model_var_type=(
            (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

    # Compatibility guard: ensure 'num_timesteps' exists for p_sample_loop(_progressive)
    # Sampling now passes a positive 'steps'; training can leave it -1 safely.
    if not hasattr(diff, "num_timesteps"):
        try:
            # prefer an internal field if present; else use steps
            val = getattr(diff, "num_diffusion_steps", None)
            if val is None:
                val = int(steps)
            diff.num_timesteps = int(val) if int(val) > 0 else 0
        except Exception:
            pass

    return diff


# ------------------------------- Argparse ------------------------------------
def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        # Skip if an option with this name already exists (prevents duplicates)
        if f"--{k}" in parser._option_string_actions:
            continue
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys, j=0):
    """
    Return a dict of key=value from an argparse namespace for the given keys.
    If a value is a list, take the j-th item.
    """
    def get_arg(k):
        arg = getattr(args, k)
        if isinstance(arg, list):
            arg = arg[j]
        return arg
    return {k: get_arg(k) for k in keys}


def str2bool(v):
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
    Two-phase parser: parse --task first for task-aware defaults, then add the rest.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["standard", "super_res", "wavelet", "curvelet"],
        default="standard",
        help="Task type.",
    )
    known_args, _ = parser.parse_known_args()
    defaults = model_and_diffusion_defaults(task=known_args.task)
    add_dict_to_argparser(parser, defaults)

    # Common script-level args (not consumed by create_model_and_diffusion)
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=100000, help="Training iterations")
    parser.add_argument("--save_interval", type=int, default=10000, help="Steps between checkpoints")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a .pt checkpoint (single-model sampling)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (sampling)")
    parser.add_argument("--cond_dir", type=str, default=None, help="Dir with conditioning images (if needed)")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--image_size", type=int, default=None, help="Full-resolution image size (sampling)")
    return parser
