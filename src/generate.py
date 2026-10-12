"""Generate images for every config point in a YAML file.

Resumable (existing images are skipped) and shardable, so long runs can be split
across several SLURM jobs:

    python -m src.generate --config configs/e2/sdxl_base.yaml --shard 0 --num-shards 4
"""

import argparse
import json
import time

import torch
import yaml

from src.common import env_info, expand_grid, load_yaml, now, read_prompts, run_dir
from src.pipelines import UNetCallCounter, build_pipeline, pipeline_key


def call_kwargs(cfg):
    kwargs = dict(cfg.get("params", {}))
    if cfg["method"] == "sdxl_lightning":
        # Each Lightning checkpoint must run with its own number of steps.
        kwargs["num_inference_steps"] = int(cfg["lightning_steps"])
    res = cfg.get("resolution")
    if res:
        kwargs.setdefault("height", res)
        kwargs.setdefault("width", res)
    return kwargs


def write_run_meta(out, cfg, resolved):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    with open(out / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({"env": env_info(), "models": resolved, "written": now()}, f, indent=2)


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
    device = "cuda"
    pipe, built_key, counter = None, None, None

    for point in expand_grid(cfg):
        if args.only and point["name"] != args.only:
            continue
        out = run_dir(point)
        prompts = read_prompts(point)
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
        write_run_meta(out, point, resolved)

        kwargs = call_kwargs(point)
        log_path = out / f"calls_shard{args.shard}of{args.num_shards}.jsonl"
        with open(log_path, "a", encoding="utf-8") as log:
            for p, seed in todo:
                counter.reset()
                generator = torch.Generator(device=device).manual_seed(int(seed))
                start = time.perf_counter()
                image = pipe(prompt=p["prompt"], generator=generator, **kwargs).images[0]
                elapsed = time.perf_counter() - start
                image.save(out / f"{p['prompt_id']}_{seed}.png")
                record = {"prompt_id": p["prompt_id"], "seed": seed, "seconds": round(elapsed, 4)}
                record.update(counter.summary())
                log.write(json.dumps(record) + "\n")
                log.flush()
        print(f"[{point['name']}] done", flush=True)


if __name__ == "__main__":
    main()
