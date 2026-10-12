"""Compute metrics for generated image folders and append them to the metrics CSV.

    python -m src.evaluate --config configs/smoke/sdxl_base.yaml --metrics clip,imagereward
    python -m src.evaluate --config configs/e1/sdxl_base.yaml --metrics clip,imagereward,fid
    python -m src.evaluate --config configs/diversity/sdxl_base.yaml --metrics lpips

All images are resized to 512x512 before scoring (docs/plan.md section 2).
FID uses data/coco_ref/<split>/ as the reference set, so the tune and eval stages
never share real-image statistics.
"""

import argparse
import itertools
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.common import (
    DATA_DIR, append_metrics, check_identity, expand_grid, load_yaml, now, read_prompts, run_dir,
)

EVAL_SIZE = 512
CLIP_MODEL = "openai/clip-vit-large-patch14"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
IMAGEREWARD_MODEL = "ImageReward-v1.0"


def load_images(out, prompts, seeds):
    items = []
    for p in prompts:
        for s in seeds:
            path = out / f"{p['prompt_id']}_{s}.png"
            if path.exists():
                items.append((p, s, path))
    return items


def to_eval_size(path):
    img = Image.open(path).convert("RGB")
    if img.size != (EVAL_SIZE, EVAL_SIZE):
        img = img.resize((EVAL_SIZE, EVAL_SIZE), Image.BICUBIC)
    return img


class ClipScorer:
    """CLIP Score = 100 * max(cos(image, text), 0); prompts truncated to 77 tokens."""

    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION).to(device).eval()
        # Pin the slow (non-"fast") processor so scores do not shift if the library default changes.
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION, use_fast=False)
        self.device = device

    @torch.no_grad()
    def __call__(self, images, texts):
        inputs = self.processor(text=texts, images=images, return_tensors="pt",
                                padding=True, truncation=True, max_length=77).to(self.device)
        img = torch.nn.functional.normalize(self.model.get_image_features(pixel_values=inputs["pixel_values"]), dim=-1)
        txt = torch.nn.functional.normalize(self.model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]), dim=-1)
        return (100 * (img * txt).sum(-1).clamp(min=0)).tolist()


class ImageRewardScorer:
    def __init__(self, device):
        _patch_transformers_for_imagereward()
        import ImageReward as RM
        self.model = RM.load(IMAGEREWARD_MODEL, device=device)

    @torch.no_grad()
    def __call__(self, images, texts):
        return [float(self.model.score(t, img)) for img, t in zip(images, texts)]


def _patch_transformers_for_imagereward():
    # ImageReward imports helpers that newer transformers moved to pytorch_utils.
    import transformers.modeling_utils as mu
    import transformers.pytorch_utils as pu
    for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices",
                 "find_pruneable_heads_and_prune_linear_layer", "prune_linear_layer"):
        if not hasattr(mu, name) and hasattr(pu, name):
            setattr(mu, name, getattr(pu, name))


def per_image_scores(scorer, items, batch_size=16):
    scores = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        scores += scorer([to_eval_size(path) for _, _, path in chunk], [p["prompt"] for p, _, _ in chunk])
    return scores


def fid(items, split, device):
    from cleanfid import fid as cleanfid
    ref = DATA_DIR / "coco_ref" / split
    if not ref.exists():
        raise FileNotFoundError(f"{ref} missing; run python -m src.prepare_data first")
    with tempfile.TemporaryDirectory() as tmp:
        for p, s, path in items:
            to_eval_size(path).save(Path(tmp) / f"{p['prompt_id']}_{s}.png")
        return float(cleanfid.compute_fid(tmp, str(ref), mode="clean", device=torch.device(device), num_workers=0))


def lpips_diversity(items, device):
    """Mean pairwise LPIPS between seeds of the same prompt, averaged over prompts."""
    import lpips
    net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    by_prompt = defaultdict(list)
    for p, _, path in items:
        by_prompt[p["prompt_id"]].append(path)
    to_tensor = lambda path: torch.from_numpy(
        np.asarray(to_eval_size(path), dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(device)
    per_prompt = []
    with torch.no_grad():
        for paths in by_prompt.values():
            if len(paths) < 2:
                continue
            tensors = [to_tensor(x) for x in paths]
            dists = [net(a, b).item() for a, b in itertools.combinations(tensors, 2)]
            per_prompt.append(float(np.mean(dists)))
    return float(np.mean(per_prompt)), len(per_prompt)


METRICS = {"clip", "imagereward", "fid", "lpips"}


def generation_hardware(out):
    """GPU(s) that generated the images, from every shard's run_info file."""
    gpus = set()
    for path in out.glob("run_info_*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                gpus.add(json.load(f)["env"]["hardware"])
        except (OSError, KeyError, ValueError):
            continue
    return "/".join(sorted(gpus)) or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", default="clip,imagereward", help="comma list of clip,imagereward,fid,lpips")
    ap.add_argument("--only", help="only the grid point with this name")
    ap.add_argument("--allow-partial", action="store_true",
                    help="score folders even if some expected images are missing")
    args = ap.parse_args()
    wanted = {m.strip() for m in args.metrics.split(",") if m.strip()}
    unknown = wanted - METRICS
    if unknown:
        ap.error(f"unknown metrics {sorted(unknown)}; choose from {sorted(METRICS)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_yaml(args.config)
    scorers = {}
    if "clip" in wanted:
        scorers["clip_score"] = ClipScorer(device)
    if "imagereward" in wanted:
        scorers["image_reward"] = ImageRewardScorer(device)

    points = [p for p in expand_grid(cfg) if not args.only or p["name"] == args.only]
    if not points:
        raise SystemExit(f"--only {args.only!r} matches no config point")
    conflicts = [c for c in (check_identity(run_dir(p), p) for p in points) if c]
    if conflicts:
        raise SystemExit("\n".join(conflicts) + "\nthese folders do not match the config; not scored")

    incomplete = []
    for point in points:
        out = run_dir(point)
        prompts = read_prompts(point)
        items = load_images(out, prompts, point["seeds"])
        expected = len(prompts) * len(point["seeds"])
        if len(items) != expected:
            print(f"[{point['name']}] {len(items)}/{expected} images found in {out}", flush=True)
            if not args.allow_partial:
                incomplete.append(point["name"])
                continue
        if not items:
            continue
        base = {"experiment": point["experiment"], "split": point["split"], "method": point["method"],
                "config": point["name"], "hardware": generation_hardware(out), "timestamp": now()}
        rows, per_image = [], {}
        for metric, scorer in scorers.items():
            values = per_image_scores(scorer, items)
            per_image[metric] = values
            rows.append({**base, "metric": metric, "value": round(float(np.mean(values)), 4), "n_samples": len(values)})
        if "fid" in wanted:
            rows.append({**base, "metric": "fid", "value": round(fid(items, point["split"], device), 4), "n_samples": len(items)})
        if "lpips" in wanted:
            value, n = lpips_diversity(items, device)
            rows.append({**base, "metric": "lpips_diversity", "value": round(value, 4), "n_samples": n})

        append_metrics(rows)
        if per_image:
            with open(out / "per_image_scores.jsonl", "w", encoding="utf-8") as f:
                for i, (p, s, _) in enumerate(items):
                    f.write(json.dumps({"prompt_id": p["prompt_id"], "seed": s,
                                        **{m: round(v[i], 4) for m, v in per_image.items()}}) + "\n")
        for r in rows:
            print(f"[{point['name']}] {r['metric']} = {r['value']} (n={r['n_samples']})", flush=True)

    if incomplete:
        raise SystemExit(f"not scored, images missing: {incomplete} (rerun generation or pass --allow-partial)")


if __name__ == "__main__":
    main()
