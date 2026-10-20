"""Run the Turbo/Lightning experiments and diversity runs through src, with auditable freezing.

Each phase runs as one SLURM job and calls the src/ modules; this module never submits
jobs from compute nodes.
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.common import REPO_ROOT, SPLITS_DIR, OUTPUTS_DIR, expand_grid, identity, read_prompts

METHODS = ("sdxl_turbo", "sdxl_lightning")
DIVERSITY_SEEDS = list(range(8))
FROZEN = REPO_ROOT / "configs" / "c_frozen"


def run(module, *args):
    command = [sys.executable, "-m", module, *map(str, args)]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def audit_splits():
    """Require the full, disjoint splits from src.prepare_data before an official experiment."""
    result, ids = {}, {}
    for split, expected in (("tune", 500), ("eval", 2000), ("diversity", 100)):
        path = SPLITS_DIR / f"{split}.jsonl"
        rows = read_prompts({"split": split})
        keys = [p["prompt_id"] for p in rows]
        image_ids = [p["image_id"] for p in rows]
        if len(rows) != expected or len(set(keys)) != expected or len(set(image_ids)) != expected:
            raise ValueError(f"{split}: expected {expected} distinct prompt/image IDs")
        ids[split] = set(image_ids)
        result[split] = {"n_prompts": expected, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if ids["tune"] & ids["eval"] or not ids["diversity"] <= ids["eval"]:
        raise ValueError("tune/eval overlap or diversity is not a subset of eval")
    return result


def write_once(path, data):
    """Never silently replace a frozen decision or its derived configuration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = yaml.safe_load(path.read_text(encoding="utf-8"))
        if old != data:
            raise ValueError(f"frozen file differs: {path}; investigate before changing a freeze")
        return
    if path.suffix == ".json":
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def freeze(selection_path):
    """Export E1 points plus the quality-priority point for diversity.

    The baseline and LCM-LoRA selection records are frozen through the same command.
    Selection happens only in src.select_configs, before final evaluation.
    """
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    quality, lowest = selection["quality"], selection["lowest_step"]
    method = quality["method"]
    if method not in (*METHODS, "sdxl_base", "lcm_lora"):
        raise ValueError(f"unsupported selection method {method}")
    if any(p["split"] != "tune" or p.get("exploratory") or p["method"] != method
           or p.get("limit") or p.get("prompts_file") for p in (quality, lowest)):
        raise ValueError("freeze requires full official tuning configs for one method")
    if selection.get("clip_tolerance") != 1.0:
        raise ValueError("freeze requires the preset CLIP tolerance of 1.0")
    if method == "lcm_lora" and (quality.get("lora_scale", 1.0) != 1.0
                                 or quality.get("lora_fuse", True) is False):
        raise ValueError("main comparison requires fused default LCM scale 1.0")
    split_info = audit_splits()
    record = FROZEN / f"selection_{method}.json"
    # On resume reuse the original timestamps/provenance, never overwrite with a
    # freshly rescored selection. If the decision changes, write_once fails.
    write_once(record, selection)
    # Write the canonical configs/e1/<method>.yaml + selection record for Turbo and
    # Lightning; the baseline and LCM-LoRA ones are already in configs/e1/.
    if method in METHODS:
        write_once(REPO_ROOT / "configs/e1" / f"selection_{method}.json", selection)
        canonical = copy.deepcopy(quality)
        canonical.update(experiment="e1", split="eval")
        canonical.pop("name", None)
        step_key = "lightning_steps" if method == "sdxl_lightning" else "num_inference_steps"
        if step_key == "lightning_steps":
            canonical.pop(step_key, None)
        else:
            canonical["params"].pop(step_key, None)
        values = sorted({p.get("lightning_steps", p.get("params", {}).get("num_inference_steps"))
                         for p in (quality, lowest)})
        canonical["grid"] = {step_key: values}
        expected = []
        for p in (quality, lowest):
            point = copy.deepcopy(p)
            point.update(experiment="e1", split="eval")
            if identity(point) not in expected:
                expected.append(identity(point))
        expanded = [identity(p) for p in expand_grid(canonical)]
        if len(expanded) != len(expected) or any(p not in expanded for p in expected):
            raise ValueError("canonical E1 grid does not match selected points")
        write_once(REPO_ROOT / "configs/e1" / f"{method}.yaml", canonical)
    points, names = [], set()
    for p in (quality, lowest):
        if p["name"] in names:
            continue
        names.add(p["name"])
        point = copy.deepcopy(p)
        point.update(experiment="e1", split="eval")
        point.pop("limit", None)
        path = FROZEN / "e1" / f"{method}_{p['name']}.yaml"
        write_once(path, point)
        points.append(path.relative_to(REPO_ROOT).as_posix())
    diversity = copy.deepcopy(quality)
    diversity.update(experiment="diversity", split="diversity", seeds=DIVERSITY_SEEDS)
    diversity.pop("limit", None)
    div_path = FROZEN / "diversity" / f"{method}.yaml"
    write_once(div_path, diversity)
    manifest = {"method": method, "selection": record.relative_to(REPO_ROOT).as_posix(),
                "e1_configs": points, "diversity_config": div_path.relative_to(REPO_ROOT).as_posix(),
                "splits": split_info, "diversity_rule": "quality-priority point; 100 prompts x seeds 0..7"}
    write_once(FROZEN / f"manifest_{method}.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def e2(method):
    audit_splits()
    config = f"configs/e2/{method}.yaml"
    # The frozen record makes a successfully finished E2 phase safely resumable.
    frozen = FROZEN / f"selection_{method}.json"
    if not frozen.exists():
        run("src.generate", "--config", config)
        run("src.evaluate", "--config", config, "--metrics", "clip,imagereward")
        run("src.select_configs", "--config", config)
        frozen = OUTPUTS_DIR / "selection" / f"e2_{method}.json"
    freeze(frozen)


def final(method, phase):
    manifest = json.loads((FROZEN / f"manifest_{method}.json").read_text(encoding="utf-8"))
    if audit_splits() != manifest["splits"]:
        raise ValueError("split content changed after freezing")
    configs = manifest["e1_configs"] if phase == "e1" else [manifest["diversity_config"]]
    for config in configs:
        run("src.generate", "--config", config)
        metrics = "clip,imagereward,fid" if phase == "e1" else "clip,imagereward,lpips"
        run("src.evaluate", "--config", config, "--metrics", metrics)
        if phase == "e1":
            run("src.timing", "--config", config)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("audit", "e2", "exploratory", "freeze", "e1", "diversity"))
    ap.add_argument("--method", choices=(*METHODS, "sdxl_base", "lcm_lora"))
    ap.add_argument("--selection", help="completed src.select_configs JSON")
    args = ap.parse_args()
    if args.phase == "audit":
        print(json.dumps(audit_splits(), indent=2))
    elif args.phase == "freeze":
        if not args.selection:
            ap.error("freeze requires --selection")
        freeze(args.selection)
    elif args.phase == "exploratory":
        audit_splits()
        config = "configs/e2/sdxl_lightning_1step.yaml"
        run("src.generate", "--config", config)
        run("src.evaluate", "--config", config, "--metrics", "clip,imagereward")
    elif args.phase == "e2":
        if args.method not in METHODS:
            ap.error("the e2 phase supports Turbo and Lightning only")
        e2(args.method)
    else:
        if not args.method:
            ap.error("--method is required")
        final(args.method, args.phase)


if __name__ == "__main__":
    main()
