# improved_diffusion/train_util.py
import os
import time
import copy
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim

# Use the safe dist wrapper we installed earlier.
from . import dist_util as dist
from . import logger


def _as_float(x, default=0.9999):
    try:
        return float(x)
    except Exception:
        return float(default)


class ModelEMA(nn.Module):
    """Exponential moving average (EMA) wrapper model."""
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        super().__init__()
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).to(device if device is not None else next(model.parameters()).device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for p_ema, p in zip(self.ema.parameters(), model.parameters()):
            p_ema.data.mul_(d).add_(p.data, alpha=1.0 - d)
        # copy buffers exactly (e.g., for GroupNorm stats)
        for b_ema, b in zip(self.ema.buffers(), model.buffers()):
            b_ema.copy_(b)

    def state_dict(self, *args, **kwargs):
        return self.ema.state_dict(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return self


class TrainLoop:
    """
    Minimal, Colab-friendly training loop for diffusion models.

    This version avoids mandatory DDP:
      * If a torch.distributed process group exists -> wraps in DDP.
      * Otherwise trains single GPU with the raw model.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        diffusion: Any,
        data,
        batch_size: int,
        microbatch: int = -1,
        lr: float = 1e-4,
        ema_rate: Any = 0.9999,
        log_interval: int = 10,
        save_interval: int = 10000,
        resume_checkpoint: str = "",
        use_fp16: bool = False,
        fp16_scale_growth: float = 1e-3,
        weight_decay: float = 0.0,
        lr_anneal_steps: int = 0,
        max_training_steps: int = 0,
        **kwargs,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = int(batch_size)
        self.microbatch = int(microbatch)
        self.lr = float(lr)
        self.log_interval = int(log_interval)
        self.save_interval = int(save_interval)
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = bool(use_fp16)
        self.fp16_scale_growth = float(fp16_scale_growth)
        self.weight_decay = float(weight_decay)
        self.lr_anneal_steps = int(lr_anneal_steps)
        self.max_training_steps = int(max_training_steps) if int(max_training_steps) > 0 else None

        self.device = dist.dev()
        self.model.to(self.device)

        # DDP ONLY IF a group is initialized; else leave as-is
        self.ddp = False
        try:
            if dist.using_distributed() and dist.get_world_size() > 1:
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.model = DDP(self.model, device_ids=[self.device.index] if self.device.type == "cuda" else None)
                self.ddp = True
        except Exception:
            self.ddp = False  # fall back gracefully

        # Configure optimizer
        if self.weight_decay > 0.0:
            self.opt = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.opt = optim.Adam(self.model.parameters(), lr=self.lr)

        # AMP (kept off by default; pass --use_fp16 True if you want it)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)

        # EMA
        self.ema_decay = _as_float(ema_rate, 0.9999)
        # store EMA over the underlying module (unwrap DDP)
        self.ema = ModelEMA(self.model.module if self.ddp else self.model, decay=self.ema_decay, device=self.device)

        # Step counter
        self.step = 0

        # For logging / saving
        self.run_dir = logger.get_dir() or "."
        os.makedirs(self.run_dir, exist_ok=True)

        # Resume if provided (optional — safe no-op if file missing)
        if self.resume_checkpoint and os.path.isfile(self.resume_checkpoint):
            try:
                state = torch.load(self.resume_checkpoint, map_location="cpu")
                (self.model.module if self.ddp else self.model).load_state_dict(state)
                logger.log(f"Resumed weights from {self.resume_checkpoint}")
            except Exception as e:
                logger.log(f"Resume failed ({e}); continuing from scratch.")

    # -------- utils --------
    def _next_batch(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a batch from data generator and move to device."""
        x, model_kwargs = next(self.data)  # generator yields (X, {"conditional": ..., ...})
        if isinstance(x, torch.Tensor):
            x = x.to(self.device, non_blocking=True)
        else:
            x = torch.as_tensor(x, device=self.device)

        kw = {}
        if isinstance(model_kwargs, dict):
            for k, v in model_kwargs.items():
                if isinstance(v, torch.Tensor):
                    kw[k] = v.to(self.device, non_blocking=True)
                else:
                    kw[k] = torch.as_tensor(v, device=self.device)
        return x, kw

    def _iter_microbatches(self, x: torch.Tensor, kw: Dict[str, torch.Tensor]):
        B = x.shape[0]
        mb = self.batch_size if self.microbatch is None or self.microbatch <= 0 else self.microbatch
        for s in range(0, B, mb):
            x_mb = x[s : s + mb]
            kw_mb = {k: (v[s : s + mb] if isinstance(v, torch.Tensor) and v.shape[:1] == (B,) else v) for k, v in kw.items()}
            yield x_mb, kw_mb

    def _lr_now(self) -> float:
        if self.lr_anneal_steps and self.step < self.lr_anneal_steps:
            frac = 1.0 - (self.step / float(self.lr_anneal_steps))
            return max(1e-8, self.lr * frac)
        return self.lr

    def _set_lr(self, lr_value: float):
        for pg in self.opt.param_groups:
            pg["lr"] = lr_value

    def _save_ckpt(self, tag: str):
        # Save raw model and EMA model (unwrap DDP)
        raw = (self.model.module if self.ddp else self.model).state_dict()
        ema = self.ema.state_dict()
        fn_raw = os.path.join(self.run_dir, f"model_{tag}.pt")
        fn_ema = os.path.join(self.run_dir, f"ema_{tag}.pt")
        try:
            torch.save(raw, fn_raw)
            torch.save(ema, fn_ema)
            logger.log(f"Saved checkpoints: {os.path.basename(fn_raw)}, {os.path.basename(fn_ema)}")
        except Exception as e:
            logger.log(f"Warning: failed to save checkpoints ({e})")

    # -------- core loop --------
    def run_loop(self):
        self.model.train()
        t0 = time.time()
        world = dist.get_world_size()

        while True:
            if self.max_training_steps is not None and self.step >= self.max_training_steps:
                logger.log("Reached max_training_steps. Exiting training loop.")
                break

            # anneal lr if requested
            lr_now = self._lr_now()
            self._set_lr(lr_now)

            # fetch a batch
            x, kw = self._next_batch()

            # sample timesteps uniformly
            num_timesteps = getattr(self.diffusion, "num_timesteps", 1000)
            t = torch.randint(0, num_timesteps, (x.shape[0],), device=self.device)

            self.opt.zero_grad(set_to_none=True)

            # microbatch
            losses_collect = []
            for x_mb, kw_mb in self._iter_microbatches(x, kw):
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    # Many diffusion libs expect: training_losses(model, x_start, t, model_kwargs)
                    losses = self.diffusion.training_losses(
                        self.model, x_mb, t[: x_mb.shape[0]], model_kwargs=kw_mb
                    )
                    # Be robust to return shape: dict or tensor
                    if isinstance(losses, dict):
                        if "loss" in losses:
                            loss_mb = losses["loss"].mean()
                        elif "mse" in losses:
                            loss_mb = losses["mse"].mean()
                        else:
                            first_tensor = next((v for v in losses.values() if torch.is_tensor(v)), None)
                            loss_mb = first_tensor.mean() if first_tensor is not None else torch.zeros([], device=self.device)
                    else:
                        loss_mb = torch.as_tensor(losses, device=self.device).mean()

                losses_collect.append(loss_mb.detach())

                if self.use_fp16:
                    self.scaler.scale(loss_mb).backward()
                else:
                    loss_mb.backward()

            # step optimizer
            if self.use_fp16:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()

            # EMA update on the (unwrapped) model
            self.ema.update(self.model.module if self.ddp else self.model)

            # logging
            self.step += 1
            if self.step % self.log_interval == 0 or self.step == 1:
                loss_mean = torch.stack([l.detach() for l in losses_collect]).mean().item()
                elapsed = time.time() - t0
                logger.log(
                    f"step {self.step} | loss {loss_mean:.6f} | lr {lr_now:.6e} | "
                    f"world {world} | time {elapsed:.1f}s"
                )

            # checkpointing
            if self.save_interval > 0 and self.step % self.save_interval == 0:
                self._save_ckpt(f"{self.step:09d}")

        # final save on exit
        self._save_ckpt(f"{self.step:09d}_final")
