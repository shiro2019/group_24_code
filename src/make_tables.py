"""Merge the per-account metrics CSVs and build the summary tables for the report.

    python -m src.make_tables outputs/metrics/metrics.csv metrics_b.csv metrics_c.csv ...

Each cluster account appends to its own outputs/metrics/metrics.csv; pass all of them.
Rows are merged on (experiment, split, method, config, metric), keeping the newest
timestamp, and smoke-test rows are dropped. Which configs go into each table and their
settings come from the files in configs/ (frozen points from configs/e1/selection_*.json),
every number from the metrics. Written to --out (default outputs/tables):

    metrics_merged.csv     the merged long table
    e1_main.csv            E1 on the eval split: frozen points of every method, the 8-step
                           baseline reference, the E4 hybrid and its no-refine control, and
                           the exploratory Turbo 1024 points; quality, latency and memory
    latency_sorted.csv     the same rows, sorted by latency
    e2_sweep.csv           every E2 tuning point, frozen points marked in `selected`
    e3_scale.csv           LCM-LoRA adapter scale: tuning sweep and eval re-tests
    e4_strength_tune.csv   E4 refine strength on the tuning split, with the no-refine control
    diversity.csv          100 prompts x 8 seeds at each method's quality point
    e5_parti.csv           quality points on the PartiPrompts subset (if E5 rows exist)

Latency for the E4 rows comes from the timing of the same configs on the tuning split
(timing does not depend on the prompts). An empty cell means the metric is missing.
"""

import argparse
import csv
import json
from pathlib import Path

from src.common import METRICS_COLUMNS, OUTPUTS_DIR, REPO_ROOT, expand_grid, load_yaml

CONFIGS = REPO_ROOT / "configs"
METHODS = {  # method id -> label used in the tables and figures, in table order
    "sdxl_base": "SDXL + DPM-Solver++",
    "lcm_lora": "LCM-LoRA",
    "sdxl_turbo": "SDXL-Turbo",
    "sdxl_lightning": "SDXL-Lightning",
}
HYBRID_LABEL = "E4 Turbo draft + SDXL refine"
TURBO_1024_LABEL = "SDXL-Turbo @1024 (exploratory)"
DIGITS = {"fid": 2, "clip_score": 2, "image_reward": 3, "lpips_diversity": 4,
          "latency_median_s": 3, "latency_iqr_s": 3, "peak_mem_gb": 1}


def merge(paths):
    """Newest row per (experiment, split, method, config, metric), without smoke rows."""
    latest = {}
    for path in paths:
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["experiment"] == "smoke":
                    continue
                key = tuple(row[k] for k in ("experiment", "split", "method", "config", "metric"))
                if key not in latest or row["timestamp"] > latest[key]["timestamp"]:
                    latest[key] = {k: row[k] for k in METRICS_COLUMNS}
    return [latest[k] for k in sorted(latest)]


class Metrics:
    def __init__(self, rows):
        self.values = {tuple(r[k] for k in ("experiment", "split", "method", "config", "metric")): r["value"]
                       for r in rows}

    def raw(self, experiment, split, method, config, metric):
        return self.values.get((experiment, split, method, config, metric), "")

    def get(self, experiment, split, method, config, metric):
        """Rounded value as text, or "" when missing."""
        value = self.raw(experiment, split, method, config, metric)
        if value == "" or metric not in DIGITS:
            return value
        return f"{float(value):.{DIGITS[metric]}f}"


def fmt(v):
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def steps_of(cfg):
    return cfg.get("lightning_steps", cfg.get("params", {}).get("num_inference_steps"))


def cfg_of(cfg):
    return cfg.get("params", {}).get("guidance_scale", 0.0)


def points(path):
    return list(expand_grid(load_yaml(CONFIGS / path))) if (CONFIGS / path).exists() else []


def selections():
    """{method: {"quality": name, "lowest-step": name, "points": (quality cfg, lowest-step cfg)}}."""
    out = {}
    for method in METHODS:
        path = CONFIGS / "e1" / f"selection_{method}.json"
        if path.exists():
            sel = json.loads(path.read_text(encoding="utf-8"))
            out[method] = {"quality": sel["quality"]["name"], "lowest-step": sel["lowest_step"]["name"],
                           "points": (sel["quality"], sel["lowest_step"])}
    return out


def quality_columns(m, experiment, split, method, name):
    return {"fid": m.get(experiment, split, method, name, "fid"),
            "clip_score": m.get(experiment, split, method, name, "clip_score"),
            "image_reward": m.get(experiment, split, method, name, "image_reward")}


def timing_columns(m, experiment, split, method, name):
    return {"latency_s": m.get(experiment, split, method, name, "latency_median_s"),
            "latency_iqr_s": m.get(experiment, split, method, name, "latency_iqr_s"),
            "peak_mem_gb": m.get(experiment, split, method, name, "peak_mem_gb")}


def e1_rows(m, sel):
    rows = []

    def single(label, method, cfg, role):
        name = cfg["name"]
        rows.append({"method": label, "config": name, "role": role,
                     "native_px": cfg.get("resolution", 1024), "steps": steps_of(cfg), "cfg": fmt(cfg_of(cfg)),
                     "unet_calls": m.get("e1", "eval", method, name, "unet_calls_per_image"),
                     "unet_batch": m.get("e1", "eval", method, name, "unet_batch_size"),
                     **quality_columns(m, "e1", "eval", method, name),
                     **timing_columns(m, "e1", "eval", method, name)})

    for method, label in METHODS.items():
        if method not in sel:
            continue
        quality, lowest = sel[method]["points"]
        single(label, method, quality, "quality")
        if method == "sdxl_base":
            for cfg in points("e1/sdxl_base_8step.yaml"):
                single(label, method, cfg, "latency-matched reference")
        if lowest["name"] != quality["name"]:
            single(label, method, lowest, "lowest-step")
    for path, role in (("e4/eval_hybrid.yaml", "hybrid (strength {})"), ("e4/eval_draft_only.yaml", "no-refine control")):
        for cfg in points(path):
            name = cfg["name"]
            draft = m.raw("e4", "tune", "hybrid", name, "draft_unet_calls")
            refine = m.raw("e4", "tune", "hybrid", name, "refine_unet_calls")
            calls = f"{draft}+{refine}" if draft and refine else ""
            refined = cfg.get("refine", True)
            draft_cfg = fmt(cfg["draft"].get("guidance_scale", 0.0))
            rows.append({"method": HYBRID_LABEL, "config": name,
                         "role": role.format(fmt(cfg["params"]["strength"])) if refined else role,
                         "native_px": cfg["resolution"], "steps": calls,
                         "cfg": f"{draft_cfg} / {fmt(cfg_of(cfg))}" if refined else draft_cfg,
                         "unet_calls": calls, "unet_batch": m.get("e4", "tune", "hybrid", name, "unet_batch_size"),
                         **quality_columns(m, "e4", "eval", "hybrid", name),
                         **timing_columns(m, "e4", "tune", "hybrid", name)})
    for cfg in sorted(points("e1/sdxl_turbo_1024.yaml"), key=steps_of, reverse=True):
        name = cfg["name"]
        rows.append({"method": TURBO_1024_LABEL, "config": name, "role": "exploratory",
                     "native_px": cfg["resolution"], "steps": steps_of(cfg), "cfg": fmt(cfg_of(cfg)),
                     "unet_calls": m.get("e1", "eval", "sdxl_turbo", name, "unet_calls_per_image"),
                     "unet_batch": m.get("e1", "eval", "sdxl_turbo", name, "unet_batch_size"),
                     **quality_columns(m, "e1", "eval", "sdxl_turbo", name),
                     **timing_columns(m, "e1", "eval", "sdxl_turbo", name)})
    return rows


def e2_rows(m, sel):
    rows = []
    files = [(method, f"e2/{method}.yaml") for method in METHODS] + [("sdxl_lightning", "e2/sdxl_lightning_1step.yaml")]
    for method, path in files:
        marks = {sel[method]["quality"]: "quality", sel[method]["lowest-step"]: "lowest-step"} if method in sel else {}
        for cfg in points(path):
            name, experiment = cfg["name"], cfg["experiment"]
            rows.append({"method": method, "config": name, "steps": steps_of(cfg), "cfg": fmt(cfg_of(cfg)),
                         "selected": "exploratory" if cfg.get("exploratory") else marks.get(name, ""),
                         "clip_score": m.get(experiment, "tune", method, name, "clip_score"),
                         "image_reward": m.get(experiment, "tune", method, name, "image_reward"),
                         "latency_s": m.get(experiment, "tune", method, name, "latency_median_s")})
    order = list(METHODS)
    return sorted(rows, key=lambda r: (order.index(r["method"]), float(r["cfg"]), r["steps"]))


def e3_rows(m):
    evals = {cfg["lora_scale"]: cfg["name"] for cfg in points("e3/eval_lcm_lora.yaml")}
    rows = []
    for cfg in points("e3/lcm_lora.yaml"):
        name, ev = cfg["name"], evals.get(cfg["lora_scale"], "")
        rows.append({"lora_scale": fmt(cfg["lora_scale"]),
                     "tune_clip": m.get("e3", "tune", "lcm_lora", name, "clip_score"),
                     "tune_ir": m.get("e3", "tune", "lcm_lora", name, "image_reward"),
                     "eval_fid": m.get("e3", "eval", "lcm_lora", ev, "fid"),
                     "eval_clip": m.get("e3", "eval", "lcm_lora", ev, "clip_score"),
                     "eval_ir": m.get("e3", "eval", "lcm_lora", ev, "image_reward")})
    return rows


def e4_rows(m):
    rows = []
    for cfg in points("e4/draft_only.yaml") + points("e4/hybrid.yaml"):
        name = cfg["name"]
        rows.append({"config": name,
                     "strength": fmt(cfg["params"]["strength"]) if cfg.get("refine", True) else "0 (no refine)",
                     "refine_unet_calls": m.get("e4", "tune", "hybrid", name, "refine_unet_calls"),
                     "clip_score": m.get("e4", "tune", "hybrid", name, "clip_score"),
                     "image_reward": m.get("e4", "tune", "hybrid", name, "image_reward"),
                     "latency_s": m.get("e4", "tune", "hybrid", name, "latency_median_s"),
                     "peak_mem_gb": m.get("e4", "tune", "hybrid", name, "peak_mem_gb")})
    return rows


def diversity_rows(m):
    rows = []
    for method, label in METHODS.items():
        for cfg in points(f"c_frozen/diversity/{method}.yaml"):
            name = cfg["name"]
            rows.append({"method": label, "config": name,
                         "lpips_diversity": m.get("diversity", "diversity", method, name, "lpips_diversity"),
                         "clip_score": m.get("diversity", "diversity", method, name, "clip_score"),
                         "image_reward": m.get("diversity", "diversity", method, name, "image_reward")})
    return rows


def e5_rows(m, merged):
    found = sorted({(r["method"], r["config"], r["n_samples"]) for r in merged
                    if r["experiment"] == "e5" and r["metric"] in ("clip_score", "image_reward")},
                   key=lambda t: (list(METHODS).index(t[0]) if t[0] in METHODS else len(METHODS), t[1]))
    return [{"method": METHODS.get(method, method), "config": name, "n_prompts": n,
             "clip_score": m.get("e5", "parti", method, name, "clip_score"),
             "image_reward": m.get("e5", "parti", method, name, "image_reward")}
            for method, name, n in found]


def write(rows, path, columns=None):
    columns = columns or (list(rows[0]) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    missing = sum(v == "" for r in rows for k, v in r.items() if k != "selected")
    print(f"  {path.name}: {len(rows)} rows" + (f", {missing} empty cells" if missing else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", nargs="+", help="metrics CSV files (the metrics.csv format)")
    ap.add_argument("--out", default=str(OUTPUTS_DIR / "tables"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged = merge(args.metrics)
    m, sel = Metrics(merged), selections()
    print(f"tables in {out}:")
    write(merged, out / "metrics_merged.csv", METRICS_COLUMNS)
    e1 = e1_rows(m, sel)
    write(e1, out / "e1_main.csv")
    write(sorted(e1, key=lambda r: float(r["latency_s"]) if r["latency_s"] else float("inf")),
          out / "latency_sorted.csv")
    write(e2_rows(m, sel), out / "e2_sweep.csv")
    write(e3_rows(m), out / "e3_scale.csv")
    write(e4_rows(m), out / "e4_strength_tune.csv")
    write(diversity_rows(m), out / "diversity.csv")
    e5 = e5_rows(m, merged)
    if e5:
        write(e5, out / "e5_parti.csv")


if __name__ == "__main__":
    main()
