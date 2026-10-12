"""Generate images for every config point in a YAML file.

Resumable (existing images are skipped) and shardable, so long runs can be split
across several SLURM jobs:

    python -m src.generate --config configs/e2/sdxl_base.yaml --shard 0 --num-shards 4
"""

import argparse
import inspect
import json
import os
import time

import torch
import yaml
from PIL import Image

from src.common import CONFIG_KEYS, check_identity, env_info, expand_grid, load_yaml, now, read_prompts, run_dir
from src.pipelines import UNetCallCounter, build_pipeline, pipeline_key

# Allowed guidance range per method (docs/plan.md section 2); None means unrestricted.
GUIDANCE_RANGE = {"sdxl_turbo": (0.0, 0.0), "sdxl_lightning": (0.0, 0.0), "lcm_lora": (0.0, 2.0)}


def call_kwargs(cfg):
    """Arguments for pipe(...), checked against the method's valid configuration."""
    params = dict(cfg.get("params", {}))
    misplaced = CONFIG_KEYS & set(params)
    if misplaced:
        raise ValueError(f"[{cfg['name']}] {sorted(misplaced)} must be top-level config keys, not params")

    method = cfg["method"]
    if method in ("sdxl_turbo", "sdxl_lightning"):
        params.setdefault("guidance_scale", 0.0)  # diffusers would otherwise default to 5.0 (CFG on)
    if method == "sdxl_lightning":
        # Each Lightning checkpoint must run with its own number of steps.
        steps = int(cfg["lightning_steps"])
        if params.get("num_inference_steps", steps) != steps:
            raise ValueError(f"[{cfg['name']}] num_inference_steps must equal lightning_steps={steps}")
        params["num_inference_steps"] = steps
    lo_hi = GUIDANCE_RANGE.get(method)
    g = params.get("guidance_scale")
    if lo_hi and g is not None and not (lo_hi[0] <= g <= lo_hi[1]) and not cfg.get("exploratory"):
        raise ValueError(f"[{cfg['name']}] guidance_scale={g} is outside {lo_hi} for {method}; "
                         "set exploratory: true to run it anyway")

    res = cfg.get("resolution")
    if res:
        params.setdefault("height", res)
        params.setdefault("width", res)
    return params


def save_image(image, out, filename, prompt_index, cfg):
    """Metrics are computed at 512px, so images are stored at save_size to fit the disk
    quota; the first full_res_first_n prompts also keep a full-resolution copy for crops.
    Files are written under a temporary name and renamed, so a killed job never leaves a
    truncated PNG that a resumed run would skip."""
    save_size = cfg.get("save_size", 512)
    resized = image if not save_size or image.size == (save_size, save_size) else image.resize(
        (save_size, save_size), Image.BICUBIC)
    if prompt_index < cfg.get("full_res_first_n", 20) and resized is not image:
        (out / "full").mkdir(exist_ok=True)
        _atomic_save(image, out / "full" / filename)
    _atomic_save(resized, out / filename)


def _atomic_save(image, path):
    tmp = path.with_name(path.name + ".tmp")
    image.save(tmp, format="PNG")
    os.replace(tmp, path)


def write_config(out, cfg):
    """Record the config; called only after check_identity() found no conflict."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    for tmp in out.glob("*.png.tmp"):  # left behind by a job killed while saving
        tmp.unlink()


def write_run_info(out, resolved, shard_tag):
    with open(out / f"run_info_{shard_tag}.json", "w", encoding="utf-8") as f:
        json.dump({"env": env_info(), "models": resolved, "written": now()}, f, indent=2)


def check_call_args(pipe, kwargs, name):
    """The SDXL pipelines accept **kwargs, so a misspelled argument would be ignored."""
    accepted = set(inspect.signature(pipe.__call__).parameters)
    unknown = set(kwargs) - accepted
    if unknown:
        raise ValueError(f"[{name}] {type(pipe).__name__} does not accept {sorted(unknown)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, help="only the first N prompts (debugging)")
    ap.add_argument("--only", help="only the grid point with this name")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.limit:
        cfg["limit"] = args.limit
    points = [p for p in expand_grid(cfg) if not args.only or p["name"] == args.only]
    if not points:
        raise SystemExit(f"--only {args.only!r} matches no config point")
    # Validate every point and its output folder before spending any GPU time.
    conflicts = [c for c in (check_identity(run_dir(p), p) for p in points) if c]
    if conflicts:
        raise SystemExit("\n".join(conflicts) + "\nuse another name or move the old folders away")
    for point in points:
        call_kwargs(point)
    shard_tag = f"shard{args.shard}of{args.num_shards}"
    device = "cuda"
    pipe, built_key, counter, resolved = None, None, None, None

    for point in points:
        out = run_dir(point)
        write_config(out, point)
        prompts = read_prompts(point)
        index = {p["prompt_id"]: i for i, p in enumerate(prompts)}
        jobs = [(p, s) for p in prompts for s in point["seeds"]][args.shard :: args.num_shards]
        todo = [(p, s) for p, s in jobs if not (out / f"{p['prompt_id']}_{s}.png").exists()]
        print(f"[{point['name']}] {len(jobs)} images in shard, {len(todo)} to generate", flush=True)
        if not todo:
            continue

        if pipeline_key(point) != built_key:
            if pipe is not None:
                counter.remove()
                del pipe
                torch.cuda.empty_cache()
            pipe, resolved = build_pipeline(point, device)
            counter = UNetCallCounter(pipe.unet)
            built_key = pipeline_key(point)
        write_run_info(out, resolved, shard_tag)

        kwargs = call_kwargs(point)
        check_call_args(pipe, kwargs, point["name"])
        log_path = out / f"calls_{shard_tag}.jsonl"
        with open(log_path, "a", encoding="utf-8") as log:
            for p, seed in todo:
                counter.reset()
                generator = torch.Generator(device=device).manual_seed(int(seed))
                # Wall-clock per call for monitoring only; reported latency comes from src.timing.
                start = time.perf_counter()
                image = pipe(prompt=p["prompt"], generator=generator, **kwargs).images[0]
                elapsed = time.perf_counter() - start
                save_image(image, out, f"{p['prompt_id']}_{seed}.png", index[p["prompt_id"]], point)
                record = {"prompt_id": p["prompt_id"], "seed": seed, "seconds": round(elapsed, 4)}
                record.update(counter.summary())
                log.write(json.dumps(record) + "\n")
                log.flush()
        print(f"[{point['name']}] done", flush=True)


if __name__ == "__main__":
    main()
