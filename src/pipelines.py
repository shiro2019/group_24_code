"""Build the diffusers pipeline of each method, with its sampler and scheduler settings."""

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download, model_info
from safetensors.torch import load_file

SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_TURBO = "stabilityai/sdxl-turbo"
LCM_LORA = "latent-consistency/lcm-lora-sdxl"
LIGHTNING = "ByteDance/SDXL-Lightning"

METHODS = ("sdxl_base", "lcm_lora", "sdxl_turbo", "sdxl_lightning")

# Keys that change how the pipeline is built; a change forces a rebuild.
PIPELINE_KEYS = ("method", "models", "lightning_steps", "lora_fuse", "lora_scale", "dtype", "timestep_spacing")


def pipeline_key(cfg):
    return tuple(str(cfg.get(k)) for k in PIPELINE_KEYS)


def _dtype(cfg):
    return {"fp16": torch.float16, "fp32": torch.float32}[cfg.get("dtype", "fp16")]


def _rev(cfg, repo):
    return (cfg.get("models") or {}).get(repo)


def build_pipeline(cfg, device="cuda"):
    """Return (pipe, resolved) where resolved records model revisions actually loaded."""
    method = cfg["method"]
    dtype = _dtype(cfg)
    resolved = {}

    if method == "sdxl_turbo":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_TURBO, torch_dtype=dtype, variant="fp16", revision=_rev(cfg, SDXL_TURBO)
        )
        resolved[SDXL_TURBO] = _resolve(SDXL_TURBO, cfg)

    elif method == "sdxl_lightning":
        steps = int(cfg["lightning_steps"])
        ckpt = (
            "sdxl_lightning_1step_unet_x0.safetensors" if steps == 1
            else f"sdxl_lightning_{steps}step_unet.safetensors"
        )
        unet_config = UNet2DConditionModel.load_config(
            SDXL_BASE, subfolder="unet", revision=_rev(cfg, SDXL_BASE)
        )
        unet = UNet2DConditionModel.from_config(unet_config).to(device, dtype)
        unet.load_state_dict(load_file(
            hf_hub_download(LIGHTNING, ckpt, revision=_rev(cfg, LIGHTNING)), device=device
        ))
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_BASE, unet=unet, torch_dtype=dtype, variant="fp16", revision=_rev(cfg, SDXL_BASE)
        )
        sched_kwargs = {"timestep_spacing": "trailing"}
        if steps == 1:
            sched_kwargs["prediction_type"] = "sample"
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, **sched_kwargs)
        resolved[SDXL_BASE] = _resolve(SDXL_BASE, cfg)
        resolved[LIGHTNING] = _resolve(LIGHTNING, cfg) + f":{ckpt}"

    elif method in ("sdxl_base", "lcm_lora"):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_BASE, torch_dtype=dtype, variant="fp16", revision=_rev(cfg, SDXL_BASE)
        )
        resolved[SDXL_BASE] = _resolve(SDXL_BASE, cfg)
        if method == "sdxl_base":
            # "trailing" makes the first step start at t=999; SDXL's default "leading"
            # spacing starts few-step sampling at a lower noise level than pure noise.
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, timestep_spacing=cfg.get("timestep_spacing", "trailing")
            )
        else:
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights(LCM_LORA, adapter_name="lcm", revision=_rev(cfg, LCM_LORA))
            scale = float(cfg.get("lora_scale", 1.0))
            if cfg.get("lora_fuse", True):
                # Scale must be applied at fuse time; call-time scale is ignored once fused.
                pipe.fuse_lora(lora_scale=scale)
            else:
                pipe.set_adapters(["lcm"], adapter_weights=[scale])
            resolved[LCM_LORA] = _resolve(LCM_LORA, cfg)
    else:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    sched = pipe.scheduler.config
    resolved["scheduler"] = {"class": type(pipe.scheduler).__name__, **{
        k: sched.get(k) for k in ("timestep_spacing", "steps_offset", "prediction_type", "algorithm_type")
        if k in sched
    }}
    return pipe, resolved


def img2img_from(pipe):
    """SDXL img2img sharing the components (and scheduler) of an already-built pipeline."""
    return StableDiffusionXLImg2ImgPipeline(**pipe.components)


def _resolve(repo, cfg):
    rev = _rev(cfg, repo)
    try:
        return model_info(repo, revision=rev).sha
    except Exception:
        return rev or "unresolved"


class UNetCallCounter:
    """Counts real UNet forward calls, their batch sizes and timesteps for one image.

    The hook keeps timesteps as tensors and converts them only in summary(), so it
    does not force a GPU sync on every UNet call.
    """

    def __init__(self, unet):
        self.calls = []
        self._handle = unet.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        sample = args[0] if args else kwargs.get("sample")
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        t = timestep.detach().flatten()[0] if torch.is_tensor(timestep) else timestep
        self.calls.append({"batch": int(sample.shape[0]), "t": t})

    def reset(self):
        self.calls = []

    def summary(self):
        to_float = lambda t: float(t.item()) if torch.is_tensor(t) else float(t)
        return {
            "unet_calls": len(self.calls),
            "unet_batch_sizes": sorted({c["batch"] for c in self.calls}),
            "timesteps": [to_float(c["t"]) for c in self.calls],
        }

    def remove(self):
        self._handle.remove()
