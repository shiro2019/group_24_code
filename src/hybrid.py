"""E4 hybrid refine: SDXL-Turbo draft -> bicubic upscale -> SDXL base img2img.

    python -m src.hybrid --config configs/e4/hybrid.yaml --shard 0 --num-shards 2
    python -m src.hybrid --config configs/e4/hybrid.yaml --timing

A config uses method: hybrid and
  draft:         Turbo call arguments (always 512 px; guidance must be 0 as for sdxl_turbo)
  params:        SDXL img2img call arguments of the refine stage (num_inference_steps,
                 guidance_scale, strength); DPM-Solver++ with timestep_spacing
  refine_seeds:  one img2img noise seed per draft seed in `seeds`, paired by position
  refine: false  keep only the upscaled draft: the no-refine control, which uses the
                 same draft and the same upscaling
Images are named <prompt_id>_<draft seed>.png, so src.evaluate scores them as usual.

Generation is resumable and shardable like src.generate. Each draft is made once and
refined at every grid point, so all strengths start from the same draft. calls_*.jsonl
logs the UNet calls, batch sizes and timesteps of each stage as actually executed;
int(num_inference_steps x strength) is only the expected refine step count.

--timing follows the src.timing protocol over the whole two-stage flow: batch size 1,
warm-up first, timed from the draft's text encoding until the refined image is
VAE-decoded, with torch.cuda.synchronize() on both sides. Both pipelines stay loaded,
so peak memory includes both models.
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.common import (
    CONFIG_KEYS, append_metrics, check_identity, expand_grid, hardware_tag, load_yaml, now,
    read_prompts, run_dir,
)
from src.generate import call_kwargs, check_call_args, save_image, write_config, write_run_info
from src.pipelines import UNetCallCounter, build_pipeline, img2img_from

DRAFT_RESOLUTION = 512

# Settings every point of one file must share, so one draft serves all of them.
SHARED_KEYS = ("experiment", "split", "prompts_file", "limit", "seeds", "refine_seeds", "refine",
               "draft", "models", "dtype", "resolution", "timestep_spacing", "exploratory")


def stage_cfg(cfg, method):
    """The config of one stage, in the form src.pipelines and src.generate expect."""
    sub = {k: v for k, v in cfg.items() if k not in ("draft", "refine", "refine_seeds", "grid")}
    sub["method"] = method
    if method == "sdxl_turbo":
        sub["params"] = dict(cfg.get("draft") or {})
        sub["resolution"] = DRAFT_RESOLUTION
        sub.pop("timestep_spacing", None)
    return sub


def refine_kwargs(cfg):
    """Arguments for the img2img call; the img2img output size follows its input image."""
    params = dict(cfg.get("params") or {})
    misplaced = CONFIG_KEYS & set(params)
    if misplaced:
        raise ValueError(f"[{cfg['name']}] {sorted(misplaced)} must be top-level config keys, not params")
    strength = params.get("strength")
    if strength is None or not 0 < strength <= 1:
        raise ValueError(f"[{cfg['name']}] params.strength must be in (0, 1], got {strength}")
    if int(params.get("num_inference_steps", 50) * strength) < 1:
        raise ValueError(f"[{cfg['name']}] int(num_inference_steps x strength) must be >= 1")
    return params


def check_points(points):
    first = points[0]
    for p in points:
        if p["method"] != "hybrid":
            raise ValueError(f"[{p['name']}] src.hybrid only runs method: hybrid")
        diff = [k for k in SHARED_KEYS if p.get(k) != first.get(k)]
        if diff:
            raise ValueError(f"[{p['name']}] grid points must share {diff}; sweep only the refine params")
        if not p.get("draft"):
            raise ValueError(f"[{p['name']}] needs a draft section (Turbo call arguments)")
        call_kwargs(stage_cfg(p, "sdxl_turbo"))
        if p.get("refine", True):
            refine_kwargs(p)
            if len(p.get("refine_seeds") or []) != len(p["seeds"]):
                raise ValueError(f"[{p['name']}] refine_seeds needs one noise seed per draft seed")
        elif p.get("params") or p.get("refine_seeds"):
            raise ValueError(f"[{p['name']}] refine: false takes no params or refine_seeds")


class Hybrid:
    """Both stages, built once; the refine stage is skipped when refine is false."""

    def __init__(self, cfg, device):
        self.device = device
        self.size = int(cfg.get("resolution", 1024))
        self.draft_pipe, draft_res = build_pipeline(stage_cfg(cfg, "sdxl_turbo"), device)
        self.refine_pipe = refine_res = None
        if cfg.get("refine", True):
            base, refine_res = build_pipeline(stage_cfg(cfg, "sdxl_base"), device)
            self.refine_pipe = img2img_from(base)
            self.refine_pipe.set_progress_bar_config(disable=True)
        self.resolved = {"draft": draft_res, "refine": refine_res}
        self.counters = {}

    def attach_counters(self):
        self.counters = {"draft": UNetCallCounter(self.draft_pipe.unet)}
        if self.refine_pipe is not None:
            self.counters["refine"] = UNetCallCounter(self.refine_pipe.unet)

    def detach_counters(self):
        for c in self.counters.values():
            c.remove()
        self.counters = {}

    def draft(self, prompt, seed, kwargs):
        """Turbo draft as a [0, 1] tensor, bicubic-upscaled in pixel space."""
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        image = self.draft_pipe(prompt=prompt, generator=generator, output_type="pt", **kwargs).images
        return F.interpolate(image.float(), size=(self.size, self.size), mode="bicubic",
                             align_corners=False).clamp(0, 1)

    def refine(self, prompt, upscaled, seed, kwargs):
        if self.refine_pipe is None:
            return upscaled
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        return self.refine_pipe(prompt=prompt, image=upscaled, generator=generator,
                                output_type="pt", **kwargs).images

    def reset_counters(self, stage=None):
        for name, c in self.counters.items():
            if stage in (None, name):
                c.reset()

    def calls(self):
        """UNet calls of both stages; a CFG call (cond + uncond in one batch of 2) counts once."""
        empty = {"unet_calls": 0, "unet_batch_sizes": [], "timesteps": []}
        d = self.counters["draft"].summary()
        r = self.counters["refine"].summary() if "refine" in self.counters else empty
        return {"unet_calls": d["unet_calls"] + r["unet_calls"],
                "draft_unet_calls": d["unet_calls"], "draft_unet_batch_sizes": d["unet_batch_sizes"],
                "draft_timesteps": d["timesteps"],
                "refine_unet_calls": r["unet_calls"], "refine_unet_batch_sizes": r["unet_batch_sizes"],
                "refine_timesteps": r["timesteps"]}


def to_pil(t):
    array = (t[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def generate(points, args, device):
    first = points[0]
    shard_tag = f"shard{args.shard}of{args.num_shards}"
    prompts = read_prompts(first)
    index = {p["prompt_id"]: i for i, p in enumerate(prompts)}
    seed_pairs = list(zip(first["seeds"], first.get("refine_seeds") or first["seeds"]))
    jobs = [(p, s, rs) for p in prompts for s, rs in seed_pairs][args.shard :: args.num_shards]

    outs = {}
    for point in points:
        out = outs[point["name"]] = run_dir(point)
        write_config(out, point)
        todo = sum(not (out / f"{p['prompt_id']}_{s}.png").exists() for p, s, _ in jobs)
        print(f"[{point['name']}] {len(jobs)} images in shard, {todo} to generate", flush=True)
    jobs = [j for j in jobs if any(not (o / f"{j[0]['prompt_id']}_{j[1]}.png").exists() for o in outs.values())]
    if not jobs:
        return

    hybrid = Hybrid(first, device)
    hybrid.attach_counters()
    draft_kwargs = call_kwargs(stage_cfg(first, "sdxl_turbo"))
    check_call_args(hybrid.draft_pipe, {**draft_kwargs, "output_type": "pt"}, first["name"])
    kwargs = {}
    for point in points:
        write_run_info(outs[point["name"]], hybrid.resolved, shard_tag)
        if hybrid.refine_pipe is not None:
            kwargs[point["name"]] = refine_kwargs(point)
            check_call_args(hybrid.refine_pipe, kwargs[point["name"]], point["name"])

    logs = {name: open(out / f"calls_{shard_tag}.jsonl", "a", encoding="utf-8") for name, out in outs.items()}
    try:
        for p, seed, refine_seed in jobs:
            filename = f"{p['prompt_id']}_{seed}.png"
            hybrid.reset_counters()
            start = time.perf_counter()
            upscaled = hybrid.draft(p["prompt"], seed, draft_kwargs)
            draft_seconds = time.perf_counter() - start
            for point in points:
                out = outs[point["name"]]
                if (out / filename).exists():
                    continue
                hybrid.reset_counters("refine")
                # Wall-clock for monitoring only; reported latency comes from --timing.
                start = time.perf_counter()
                image = hybrid.refine(p["prompt"], upscaled, refine_seed, kwargs.get(point["name"]))
                refine_seconds = time.perf_counter() - start
                save_image(to_pil(image), out, filename, index[p["prompt_id"]], point)
                record = {"prompt_id": p["prompt_id"], "seed": seed, "refine_seed": refine_seed,
                          "seconds": round(draft_seconds + refine_seconds, 4),
                          "draft_seconds": round(draft_seconds, 4), **hybrid.calls()}
                logs[point["name"]].write(json.dumps(record) + "\n")
                logs[point["name"]].flush()
    finally:
        for f in logs.values():
            f.close()
    print("done", flush=True)


def timing(points, args, device):
    first = points[0]
    hybrid = Hybrid(first, device)
    draft_kwargs = call_kwargs(stage_cfg(first, "sdxl_turbo"))
    prompts = read_prompts(first)
    seed, refine_seed = first["seeds"][0], (first.get("refine_seeds") or first["seeds"])[0]

    for point in points:
        kwargs = refine_kwargs(point) if hybrid.refine_pipe is not None else None

        def run(i):
            p = prompts[i % len(prompts)]
            torch.cuda.synchronize()
            start = time.perf_counter()
            hybrid.refine(p["prompt"], hybrid.draft(p["prompt"], seed, draft_kwargs), refine_seed, kwargs)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        for i in range(args.warmup):
            run(i)
        # Count UNet calls in one separate, untimed run, as src.timing does.
        hybrid.attach_counters()
        run(0)
        calls = hybrid.calls()
        hybrid.detach_counters()

        torch.cuda.reset_peak_memory_stats()
        times = [run(i) for i in range(args.runs)]
        q1, med, q3 = np.percentile(times, [25, 50, 75])
        batch_sizes = sorted(set(calls["draft_unet_batch_sizes"]) | set(calls["refine_unet_batch_sizes"]))
        base = {"experiment": point["experiment"], "split": point["split"], "method": point["method"],
                "config": point["name"], "hardware": hardware_tag(), "timestamp": now(), "n_samples": args.runs}
        rows = [
            {**base, "metric": "latency_median_s", "value": round(med, 4)},
            {**base, "metric": "latency_iqr_s", "value": round(q3 - q1, 4)},
            {**base, "metric": "peak_mem_gb", "value": round(torch.cuda.max_memory_allocated() / 2**30, 3)},
            {**base, "metric": "unet_calls_per_image", "value": calls["unet_calls"]},
            {**base, "metric": "unet_batch_size", "value": "/".join(map(str, batch_sizes))},
            {**base, "metric": "draft_unet_calls", "value": calls["draft_unet_calls"]},
            {**base, "metric": "refine_unet_calls", "value": calls["refine_unet_calls"]},
        ]
        append_metrics(rows)
        print(f"[{point['name']}] median {med:.3f}s, IQR {q3 - q1:.3f}s, UNet calls/image "
              f"{calls['unet_calls']} (draft {calls['draft_unet_calls']}, refine {calls['refine_unet_calls']} "
              f"at t={calls['refine_timesteps']}, batch {batch_sizes})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, help="only the first N prompts (debugging)")
    ap.add_argument("--only", help="only the grid point with this name")
    ap.add_argument("--timing", action="store_true", help="measure latency instead of generating")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.limit:
        cfg["limit"] = args.limit
    points = [p for p in expand_grid(cfg) if not args.only or p["name"] == args.only]
    if not points:
        raise SystemExit(f"--only {args.only!r} matches no config point")
    check_points(points)
    if args.timing:
        timing(points, args, "cuda")
        return
    # Validate every output folder before spending any GPU time.
    conflicts = [c for c in (check_identity(run_dir(p), p) for p in points) if c]
    if conflicts:
        raise SystemExit("\n".join(conflicts) + "\nuse another name or move the old folders away")
    generate(points, args, "cuda")


if __name__ == "__main__":
    main()
