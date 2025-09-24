import argparse

from . import gaussian_diffusion as gd
from.resample import ScheduleSampler
from .unet import SuperResModel, UNetModel, ConditionalModel

NUM_CLASSES = 1000


""" 
In all this file, the argument `task` represent the task the model performs. It can be:
- "standard": unconditional denoising
- "super_res": super-resolution, i.e., denoising conditioned on low-resolution
- "wavelet": denoising of high-frequencies conditioned on low-frequencies
"""


def model_and_diffusion_defaults(task="standard"):
    """
    Defaults for image training.
    """
    return dict(
        large_size=256 if task == "super_res" else 64,
        small_size=64 if task == "super_res" else 64,
        j=0,  # Scale of wavelet coefficients to use, if task == "wavelet"
        conditional=True,  # Whether to sample low or high frequencies, if task == "wavelet"
        channel_mult=None,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=-1,
        final_time=5.,
        noise_schedule="quadratic",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        learn_potential=False,
    )


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
):
    model = create_model(
        task=task,
        large_size=large_size,
        small_size=small_size,
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


def create_model(
    task,
    large_size,
    small_size,
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
        channel_mult = tuple(map(int, channel_mult.split(',')))  # Convert "123" to (1, 2, 3).

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    if task == "standard":
        model = UNetModel
    elif task == "super_res":
        model = SuperResModel
    elif task == "wavelet":
        model = ConditionalModel if conditional else UNetModel
    else:
        assert False

    kwargs = dict(
        in_channels=3,
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
    if task == "wavelet" and conditional:
        kwargs.update(in_channels=9, conditioning_channels=3)
    kwargs.update(out_channels=kwargs["in_channels"] * (2 if learn_sigma else 1))

    return model(**kwargs)


def create_gaussian_diffusion(
    *,
    steps=-1,
    final_time=5.,
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
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys, j=0):
    """ Returns a dictionary of key=value from arguments and an iterable of keys.
    Extracts the j-th item from list-valued arguments (nargs='+'). """
    def get_arg(k):
        arg = getattr(args, k)
        if isinstance(arg, list):
            arg = arg[j]
        return arg
    return {k: get_arg(k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
