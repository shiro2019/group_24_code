"""Shared helpers: paths, config loading, run metadata and the metrics CSV."""

import csv
import itertools
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = REPO_ROOT / "outputs"
METRICS_CSV = OUTPUTS_DIR / "metrics" / "metrics.csv"

METRICS_COLUMNS = [
    "experiment", "split", "method", "config", "metric",
    "value", "n_samples", "hardware", "timestamp",
]


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_grid(cfg):
    """Yield one concrete config per point of cfg['grid'] (or cfg itself if no grid).

    Grid keys override cfg['params'] (sampling arguments) unless they are listed in
    cfg['pipeline_keys'], in which case they override top-level keys such as
    'lightning_steps' or 'lora_scale' that change how the pipeline is built.
    """
    grid = cfg.get("grid") or {}
    base = {k: v for k, v in cfg.items() if k != "grid"}
    if not grid:
        base.setdefault("name", config_name(base))
        yield base
        return
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        point = json.loads(json.dumps(base))
        point.setdefault("params", {})
        for k, v in zip(keys, values):
            if k in point.get("pipeline_keys", []):
                point[k] = v
            else:
                point["params"][k] = v
        point["name"] = config_name(point, keys)
        yield point


def config_name(cfg, keys=None):
    if cfg.get("name") and not keys:
        return cfg["name"]
    params = cfg.get("params", {})
    keys = keys or sorted(params)
    abbrev = {
        "num_inference_steps": "steps", "guidance_scale": "cfg",
        "lightning_steps": "ckpt", "lora_scale": "scale", "strength": "str",
    }
    parts = []
    for k in keys:
        v = cfg[k] if k in cfg and k not in params else params.get(k)
        parts.append(f"{abbrev.get(k, k)}{v}")
    return "_".join(parts) or "default"


def read_prompts(cfg):
    """Prompts come from data/splits/<split>.jsonl unless cfg['prompts_file'] is set."""
    path = cfg.get("prompts_file") or SPLITS_DIR / f"{cfg['split']}.jsonl"
    path = REPO_ROOT / path if not Path(path).is_absolute() else Path(path)
    with open(path, encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    if cfg.get("limit"):
        prompts = prompts[: cfg["limit"]]
    return prompts


def run_dir(cfg):
    return OUTPUTS_DIR / cfg["experiment"] / cfg["split"] / cfg["method"] / cfg["name"]


def hardware_tag():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return platform.processor() or "cpu"


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def env_info():
    info = {"python": platform.python_version(), "git_commit": git_commit(), "hardware": hardware_tag()}
    for pkg in ("torch", "diffusers", "transformers", "peft", "accelerate"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            info[pkg] = None
    return info


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_metrics(rows, path=METRICS_CSV):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_COLUMNS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in METRICS_COLUMNS})
