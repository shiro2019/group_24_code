"""Measure end-to-end latency and peak memory for each config point.

    python -m src.timing --config configs/e1/sdxl_base.yaml

Protocol (docs/plan.md section 4): batch size 1, fp16, warm-up first, timed from the
pipeline call (text encoding) until the VAE-decoded tensor is ready, with
torch.cuda.synchronize() on both sides. Run all methods on the same GPU type.
"""

import argparse
import time

import numpy as np
import torch

from src.common import append_metrics, expand_grid, hardware_tag, load_yaml, now, read_prompts
from src.generate import call_kwargs
from src.pipelines import UNetCallCounter, build_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--only", help="only the grid point with this name")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda"
    for point in expand_grid(cfg):
        if args.only and point["name"] != args.only:
            continue
        pipe, _ = build_pipeline(point, device)
        counter = UNetCallCounter(pipe.unet)
        prompts = read_prompts(point)
        kwargs = call_kwargs(point)
        kwargs["output_type"] = "pt"

        def run(i):
            p = prompts[i % len(prompts)]
            generator = torch.Generator(device=device).manual_seed(int(point["seeds"][0]))
            torch.cuda.synchronize()
            start = time.perf_counter()
            pipe(prompt=p["prompt"], generator=generator, **kwargs)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        for i in range(args.warmup):
            run(i)
        torch.cuda.reset_peak_memory_stats()
        counter.reset()
        times = [run(i) for i in range(args.runs)]
        calls_per_image = len(counter.calls) / args.runs
        batch_sizes = sorted({c["batch"] for c in counter.calls})
        q1, med, q3 = np.percentile(times, [25, 50, 75])

        base = {"experiment": point["experiment"], "split": point["split"], "method": point["method"],
                "config": point["name"], "hardware": hardware_tag(), "timestamp": now(), "n_samples": args.runs}
        rows = [
            {**base, "metric": "latency_median_s", "value": round(med, 4)},
            {**base, "metric": "latency_iqr_s", "value": round(q3 - q1, 4)},
            {**base, "metric": "peak_mem_gb", "value": round(torch.cuda.max_memory_allocated() / 2**30, 3)},
            {**base, "metric": "unet_calls_per_image", "value": calls_per_image},
            {**base, "metric": "unet_batch_size", "value": "/".join(map(str, batch_sizes))},
        ]
        append_metrics(rows)
        print(f"[{point['name']}] median {med:.3f}s, IQR {q3 - q1:.3f}s, "
              f"UNet calls/image {calls_per_image:g} (batch {batch_sizes})", flush=True)

        counter.remove()
        del pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
