"""Export measured C metrics and completion checks; never invent missing results."""

import csv
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from src.common import (
    METRICS_CSV, OUTPUTS_DIR, REPO_ROOT, check_identity, expand_grid,
    load_yaml, now, read_prompts, run_dir,
)
from roles.c.workflow import FROZEN


def latest_rows():
    latest = {}
    if not METRICS_CSV.exists():
        return []
    with METRICS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] not in ("sdxl_turbo", "sdxl_lightning", "sdxl_base", "lcm_lora"):
                continue
            if row["experiment"] not in ("e2", "e2_exploratory", "e1", "diversity"):
                continue
            key = tuple(row[k] for k in ("experiment", "split", "method", "config", "metric"))
            if key not in latest or row["timestamp"] >= latest[key]["timestamp"]:
                latest[key] = row
    return list(latest.values())


def grid(cfg, destination):
    """Pre-fixed random prompt indices; all eight seeds, no image cherry-picking."""
    prompts = read_prompts(cfg)
    indices = random.Random(2026).sample(range(len(prompts)), min(4, len(prompts)))
    seeds = cfg["seeds"]
    tile, gap, label = 192, 6, 30
    canvas = Image.new("RGB", (len(seeds) * (tile + gap), len(indices) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    selected = []
    for row, index in enumerate(indices):
        prompt = prompts[index]
        selected.append(prompt)
        for col, seed in enumerate(seeds):
            path = run_dir(cfg) / f"{prompt['prompt_id']}_{seed}.png"
            if not path.exists():
                raise FileNotFoundError(path)
            with Image.open(path) as img:
                thumb = img.convert("RGB").resize((tile, tile), Image.Resampling.BICUBIC)
                canvas.paste(thumb, (col * (tile + gap), row * (tile + label)))
            draw.text((col * (tile + gap), row * (tile + label) + tile + 3),
                      f"{prompt['prompt_id']} / seed {seed}", fill="black")
    canvas.save(destination)
    destination.with_suffix(".json").write_text(json.dumps({
        "random_seed": 2026, "prompt_indices": indices, "prompts": selected,
        "configuration": cfg, "note": "Fixed random sample; resized thumbnails, not native-resolution crops."
    }, indent=2), encoding="utf-8")
    # Also export one identical random prompt across methods as the original saved
    # PNGs, for side-by-side figures without copying the full image corpus.
    example_dir = destination.parent / "example"
    example_dir.mkdir(exist_ok=True)
    manifest_path = example_dir / "manifest.json"
    first = selected[0]
    manifest = {"prompt_id": first["prompt_id"], "prompt": first["prompt"],
                "random_seed": 2026, "prompt_index": indices[0], "seeds": seeds, "methods": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["prompt_id"] != first["prompt_id"] or manifest["seeds"] != seeds:
            raise ValueError("exported example differs across methods")
    for seed in seeds:
        shutil.copyfile(run_dir(cfg) / f"{first['prompt_id']}_{seed}.png",
                        example_dir / f"{cfg['method']}_{seed}.png")
    manifest["methods"][cfg["method"]] = cfg
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def sweep_artifacts(points, rows, destination):
    """Use complete main E2 points only; exploratory one-step remains separate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method = points[0]["method"]
    score = {(r["config"], r["metric"]): float(r["value"]) for r in rows
             if r["experiment"] == "e2" and r["method"] == method}
    steps = [p.get("lightning_steps", p.get("params", {}).get("num_inference_steps")) for p in points]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, metric, title in zip(axes, ("clip_score", "image_reward"), ("CLIP Score", "ImageReward")):
        ax.plot(steps, [score[(p["name"], metric)] for p in points], "o-")
        ax.set(title=title, xlabel="Checkpoint steps" if method == "sdxl_lightning" else "Inference steps",
               xticks=steps)
        ax.grid(alpha=0.25)
    fig.suptitle(method + " / 500 tuning prompts" +
                 (" / different checkpoints" if method == "sdxl_lightning" else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(destination / f"e2_{method}_scores.png", dpi=180)
    fig.savefig(destination / f"e2_{method}_scores.svg")
    plt.close(fig)
    prompts = read_prompts(points[0])
    indices = random.Random(2026).sample(range(len(prompts)), 4)
    tile, gap, label = 192, 6, 32
    canvas = Image.new("RGB", (len(points) * (tile + gap), len(indices) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, index in enumerate(indices):
        p = prompts[index]
        for col, cfg in enumerate(points):
            with Image.open(run_dir(cfg) / f"{p['prompt_id']}_{cfg['seeds'][0]}.png") as img:
                canvas.paste(img.convert("RGB").resize((tile, tile), Image.Resampling.BICUBIC),
                             (col * (tile + gap), row * (tile + label)))
            draw.text((col * (tile + gap), row * (tile + label) + tile + 2),
                      f"{p['prompt_id']} / {cfg['name']}", fill="black")
    canvas.save(destination / f"e2_{method}_grid.png")
    (destination / f"e2_{method}_grid.json").write_text(json.dumps({
        "random_seed": 2026, "prompt_indices": indices, "prompts": [prompts[i] for i in indices],
        "configurations": points, "note": "Fixed random sample, seed 0; resized thumbnails."
    }, indent=2), encoding="utf-8")


def main():
    out = OUTPUTS_DIR / "c_handoff"
    out.mkdir(parents=True, exist_ok=True)
    rows = latest_rows()
    with (out / "metrics_latest.csv").open("w", newline="", encoding="utf-8") as f:
        columns = ["experiment", "split", "method", "config", "metric", "value", "n_samples", "hardware", "timestamp"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lookup = {(r["experiment"], r["split"], r["method"], r["config"], r["metric"]): r for r in rows}
    configs = [REPO_ROOT / "configs/e2" / f"{m}.yaml" for m in
               ("sdxl_turbo", "sdxl_lightning", "sdxl_lightning_1step")]
    configs += sorted((FROZEN / "e1").glob("*.yaml"))
    configs += sorted((FROZEN / "diversity").glob("*.yaml"))
    status, complete_e2, measured = [], {}, []
    for path in configs:
        for cfg in expand_grid(load_yaml(path)):
            # The baseline and LCM-LoRA freezes also describe their E1 points, which
            # run from their own configs/e1/ files; only their diversity runs here.
            if cfg["experiment"] == "e1" and cfg["method"] not in ("sdxl_turbo", "sdxl_lightning"):
                continue
            prompts_path = REPO_ROOT / "data/splits" / f"{cfg['split']}.jsonl"
            if not prompts_path.exists():
                status.append({"config_file": str(path.relative_to(REPO_ROOT)), "name": cfg["name"],
                               "complete": False, "reason": "prompt split missing"})
                continue
            prompts = read_prompts(cfg)
            folder = run_dir(cfg)
            files = [folder / f"{p['prompt_id']}_{s}.png" for p in prompts for s in cfg["seeds"]]
            found = sum(p.exists() for p in files)
            wanted = ["clip_score", "image_reward"]
            if cfg["experiment"] == "e1":
                wanted += ["fid"]
            if cfg["split"] == "diversity":
                wanted += ["lpips_diversity"]
            timed = cfg["experiment"] in ("e2", "e2_exploratory", "e1")
            timing_metrics = ["latency_median_s", "latency_iqr_s", "peak_mem_gb"] if timed else []
            wanted += timing_metrics
            missing = []
            scored = {}
            for metric in wanted:
                row = lookup.get((cfg["experiment"], cfg["split"], cfg["method"], cfg["name"], metric))
                n = 20 if metric in timing_metrics else (len(prompts) if metric == "lpips_diversity" else len(files))
                if row is None or int(row["n_samples"]) != n:
                    missing.append(metric)
                else:
                    scored[metric] = row
            conflict = check_identity(folder, cfg)
            complete = found == len(files) and not missing and not conflict
            item = {"config_file": str(path.relative_to(REPO_ROOT)), "name": cfg["name"],
                    "images_found": found, "images_expected": len(files), "missing_full_metrics": missing,
                    "config_conflict": conflict, "complete": complete}
            status.append(item)
            if complete:
                measured.append({"config": cfg, "metrics": scored, "n_prompts": len(prompts),
                                 "n_images": found, "config_file": str(path.relative_to(REPO_ROOT))})
            if complete and cfg["experiment"] == "e2":
                complete_e2.setdefault(cfg["method"], []).append(cfg)
            if complete and cfg["split"] == "diversity":
                grid(cfg, out / f"diversity_{cfg['method']}.png")
    for method, points in complete_e2.items():
        required = 4 if method == "sdxl_turbo" else 3
        if len(points) == required:
            sweep_artifacts(points, rows, out)
    pending = [m for m in ("sdxl_turbo", "sdxl_lightning", "sdxl_base", "lcm_lora")
               if not (FROZEN / f"manifest_{m}.json").exists()]
    complete = not pending and bool(status) and all(s["complete"] for s in status)
    report = {"generated_at": now(), "scope_complete": complete,
              "pending_freezes": pending, "configurations": status,
              "note": "Latency rows come from src.timing; the hardware column records the GPU. CSV contains measured rows only."}
    (out / "status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "results.json").write_text(json.dumps({**report, "measurements": measured}, indent=2), encoding="utf-8")
    lines = ["# Turbo, Lightning and diversity summary", "", f"Generated: {now()}", "",
             "| Configuration | Images | Required metrics complete |", "| --- | --- | --- |"]
    for item in status:
        count = f"{item.get('images_found', 0)}/{item.get('images_expected', '?')}"
        lines.append(f"| {item['config_file']} / {item['name']} | {count} | {item['complete']} |")
    lines += ["", "Missing freeze records: " + (", ".join(pending) or "none"), "",
              "Numbers are in metrics_latest.csv; incomplete configurations must not be treated as final results.",
              "Latency comes from src.timing; compare it only across runs on the same GPU type."]
    for experiment, title, metrics in (
        ("e2", "E2 tuning (500 prompts per point)", ["clip_score", "image_reward", "latency_median_s"]),
        ("e2_exploratory", "Exploratory one-step Lightning (excluded from selection)", ["clip_score", "image_reward", "latency_median_s"]),
        ("e1", "Frozen E1 (2,000 prompts per point)", ["clip_score", "image_reward", "fid", "latency_median_s"]),
        ("diversity", "Diversity (100 prompts x 8 seeds per method)", ["lpips_diversity", "clip_score", "image_reward"]),
    ):
        subset = [m for m in measured if m["config"]["experiment"] == experiment]
        if not subset:
            continue
        lines += ["", "## " + title, "", "| Method | Config | Native px | " + " | ".join(metrics) + " |",
                  "| --- | --- | ---: | " + " | ".join("---:" for _ in metrics) + " |"]
        for m in subset:
            cfg = m["config"]
            values = [m["metrics"][metric]["value"] for metric in metrics]
            lines.append(f"| {cfg['method']} | {cfg['name']} | {cfg['resolution']} | " + " | ".join(values) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
