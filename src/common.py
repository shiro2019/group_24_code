"""Shared helpers: paths, config loading, run metadata and the metrics CSV."""

import csv
import itertools
import json
import platform
import re
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


# Top-level keys that change how a pipeline is built or how images are produced.
TOP_LEVEL_KEYS = ("lightning_steps", "lora_fuse", "lora_scale", "resolution", "timestep_spacing")

# E4 two-stage configs only (method hybrid, run by src.hybrid): Turbo draft call
# arguments, the img2img noise seeds paired with `seeds`, and the no-refine switch.
HYBRID_KEYS = ("draft", "refine_seeds", "refine")

# Every key a config may use at the top level. A grid over any of these sets the
# top-level value; every other grid key is a pipeline call argument (cfg['params']).
CONFIG_KEYS = frozenset({
    *TOP_LEVEL_KEYS, "experiment", "split", "prompts_file", "limit", "seeds", "method",
    "models", "dtype", "name", "save_size", "full_res_first_n", "exploratory", "params", "grid",
    *HYBRID_KEYS,
})

# Fields that define what an output folder contains; generating or scoring a folder
# whose config.yaml differs in any of them is refused.
IDENTITY_KEYS = (
    "method", "models", "dtype", "split", "prompts_file", "save_size",
    *TOP_LEVEL_KEYS, "params", *HYBRID_KEYS,
)

NATIVE_RESOLUTION = {"sdxl_turbo": 512}

# Methods whose DPM-Solver++ stage takes timestep_spacing (hybrid: the refine stage).
SPACING_METHODS = ("sdxl_base", "hybrid")

_NAME_ORDER = (
    ("lightning_steps", "ckpt"), ("num_inference_steps", "steps"),
    ("guidance_scale", "cfg"), ("strength", "str"), ("lora_scale", "scale"),
)


def expand_grid(cfg):
    """Yield one concrete config per point of cfg['grid'] (or cfg itself if no grid)."""
    unknown = set(cfg) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}; see configs/README.md")
    if "timestep_spacing" in cfg and cfg.get("method") not in SPACING_METHODS:
        raise ValueError(f"timestep_spacing only applies to methods {SPACING_METHODS}")
    grid = cfg.get("grid") or {}
    base = {k: v for k, v in cfg.items() if k != "grid"}
    if base.get("method") in SPACING_METHODS and base.get("refine", True):
        base.setdefault("timestep_spacing", "trailing")
    if not grid:
        base.setdefault("name", config_name(base))
        yield base
        return

    keys = list(grid)
    points = []
    for values in itertools.product(*(grid[k] for k in keys)):
        point = json.loads(json.dumps(base))
        point.setdefault("params", {})
        for k, v in zip(keys, values):
            if k in ("params", "grid", "name"):
                raise ValueError(f"cannot sweep over {k!r}")
            if k in CONFIG_KEYS:
                point[k] = v
            else:
                point["params"][k] = v
        point.pop("name", None)  # grid points are always named from their settings
        point["name"] = config_name(point, swept=keys)
        points.append(point)
    names = [p["name"] for p in points]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"grid points share folder names {duplicates}")
    yield from points


def _fmt(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        text = f"{v:g}"
    elif isinstance(v, (list, tuple)):
        text = "-".join(_fmt(x) for x in v)
    else:
        text = str(v)
    return re.sub(r"[^A-Za-z0-9.+-]+", "-", text).strip("-")


def config_name(cfg, swept=()):
    """Folder name: an explicit 'name' wins for single configs; otherwise it is built from
    the sampling settings, non-default pipeline settings and every swept key, so two
    points of the same grid never share a folder."""
    if cfg.get("name"):
        return cfg["name"]
    params = cfg.get("params", {})
    merged = {**params, **{k: cfg[k] for k in TOP_LEVEL_KEYS if k in cfg}}
    parts = [f"{short}{_fmt(merged[k])}" for k, short in _NAME_ORDER if k in merged]
    parts += [f"{k}{_fmt(params[k])}" for k in sorted(params) if k not in dict(_NAME_ORDER)]
    if cfg.get("lora_fuse") is False:
        parts.append("unfused")
    native = NATIVE_RESOLUTION.get(cfg.get("method"), 1024)
    if cfg.get("resolution") and cfg["resolution"] != native:
        parts.append(f"res{cfg['resolution']}")
    covered = {k for k, _ in _NAME_ORDER} | set(params) | {"lora_fuse", "resolution"}
    parts += [f"{k}{_fmt(cfg[k])}" for k in swept if k not in covered]
    return "_".join(parts) or "default"


def identity(cfg):
    return {k: cfg.get(k) for k in IDENTITY_KEYS}


def check_identity(out, cfg):
    """Return a description of how out/config.yaml differs from cfg, or None if it
    matches or does not exist yet."""
    path = Path(out) / "config.yaml"
    if not path.exists():
        return None
    old, new = identity(load_yaml(path)), identity(cfg)
    diff = {k: (old[k], new[k]) for k in old if old[k] != new[k]}
    return f"{out} was made with different settings {diff}" if diff else None


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
