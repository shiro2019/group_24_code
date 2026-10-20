"""Apply the config selection rule to tuning-split metrics.

    python -m src.select_configs --config configs/e2/sdxl_base.yaml

Meant for the E2 tuning grids. For every config point of the file:
  1. quality point: the highest ImageReward;
  2. CLIP check: if its CLIP Score is more than --clip-tolerance below the best CLIP Score
     of the method, move on to the next ImageReward rank;
  3. lowest-step point: the fewest steps whose other settings equal the quality point's.
The newest metrics row of each config is used, and only if it was computed on the full
set of images. Exploratory configs must live in their own file and are never selected.
The selection, with the candidate table and provenance, is printed and written to
outputs/selection/<experiment>_<method>.json, ready to be frozen into the E1 configs.
"""

import argparse
import csv
import json

from src.common import METRICS_CSV, OUTPUTS_DIR, expand_grid, git_commit, load_yaml, now, read_prompts

STEP_KEYS = ("num_inference_steps", "lightning_steps")
SELECTION_METRICS = ("image_reward", "clip_score")


def latest_metrics(path, experiment, split, method):
    """{config name: {metric: row}} using the newest row per (config, metric)."""
    latest = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["experiment"], row["split"], row["method"]) != (experiment, split, method):
                continue
            if row["metric"] not in SELECTION_METRICS:
                continue
            key = (row["config"], row["metric"])
            if key not in latest or row["timestamp"] >= latest[key]["timestamp"]:
                latest[key] = row
    table = {}
    for (config, metric), row in latest.items():
        table.setdefault(config, {})[metric] = row
    return table


def steps(point):
    if point["method"] == "sdxl_lightning":
        return int(point["lightning_steps"])
    if "num_inference_steps" not in point.get("params", {}):
        raise SystemExit(f"[{point['name']}] has no num_inference_steps; cannot rank by steps")
    return int(point["params"]["num_inference_steps"])


def settings_except_steps(point):
    params = {k: v for k, v in point.get("params", {}).items() if k not in STEP_KEYS}
    top = {k: point.get(k) for k in ("lora_scale", "lora_fuse", "resolution", "timestep_spacing")}
    return json.dumps({"params": params, **top}, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the E2 tuning config whose grid was run")
    ap.add_argument("--metrics-csv", default=str(METRICS_CSV))
    ap.add_argument("--clip-tolerance", type=float, default=1.0, help="CLIP Score points, 0-100 scale")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if cfg.get("exploratory"):
        raise SystemExit("this config is exploratory; exploratory points are never selected")
    points = list(expand_grid(cfg))
    first = points[0]
    if first["split"] != "tune":
        raise SystemExit(f"selection uses the tuning split only, got split={first['split']!r}")
    if first["method"] == "lcm_lora" and any(
            p.get("lora_scale", 1.0) != 1.0 or p.get("lora_fuse", True) is False for p in points):
        print("warning: E1 uses the fused LCM-LoRA at scale 1.0; points with other settings "
              "are only a tuned variant and are not used for E1")

    table = latest_metrics(args.metrics_csv, first["experiment"], first["split"], first["method"])
    problems = []
    for p in points:
        rows = table.get(p["name"], {})
        expected = len(read_prompts(p)) * len(p["seeds"])
        for metric in SELECTION_METRICS:
            if metric not in rows:
                problems.append(f"{p['name']}: no {metric}")
            elif int(rows[metric]["n_samples"]) != expected:
                problems.append(f"{p['name']}: {metric} scored on {rows[metric]['n_samples']}/{expected} images")
    if problems:
        raise SystemExit("cannot select, rescore first:\n  " + "\n  ".join(problems))

    value = lambda p, m: float(table[p["name"]][m]["value"])
    best_clip = max(value(p, "clip_score") for p in points)
    ranked = sorted(points, key=lambda p: value(p, "image_reward"), reverse=True)
    quality = next(p for p in ranked if value(p, "clip_score") >= best_clip - args.clip_tolerance)
    same = [p for p in points if settings_except_steps(p) == settings_except_steps(quality)]
    lowest = min(same, key=steps)

    candidates = []
    print(f"{'config':<28}{'steps':>6}{'ImageReward':>13}{'CLIP':>9}{'n':>6}  scored at")
    for p in sorted(points, key=lambda p: (settings_except_steps(p), steps(p))):
        rows = table[p["name"]]
        mark = " <- quality" if p is quality else ""
        mark += " <- lowest-step" if p is lowest else ""
        print(f"{p['name']:<28}{steps(p):>6}{value(p, 'image_reward'):>13.4f}{value(p, 'clip_score'):>9.3f}"
              f"{rows['image_reward']['n_samples']:>6}  {rows['image_reward']['timestamp']}{mark}")
        candidates.append({"name": p["name"], "steps": steps(p),
                           **{m: {k: rows[m][k] for k in ("value", "n_samples", "timestamp")}
                              for m in SELECTION_METRICS}})
    print(f"best CLIP Score {best_clip:.3f}; CLIP tolerance {args.clip_tolerance}")

    out = OUTPUTS_DIR / "selection" / f"{first['experiment']}_{first['method']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"source_config": args.config, "selected_at": now(), "git_commit": git_commit(),
                   "clip_tolerance": args.clip_tolerance, "best_clip_score": best_clip,
                   "quality": quality, "lowest_step": lowest, "candidates": candidates}, f, indent=2)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
