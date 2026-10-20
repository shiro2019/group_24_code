"""Paired bootstrap confidence intervals for the per-image metrics of src.evaluate.

    python -m src.bootstrap_ci --ref base50=outputs/e1/eval/sdxl_base/steps50_cfg7.5/per_image_scores.jsonl \
        turbo4=<...>/per_image_scores.jsonl base8=<...>/per_image_scores.jsonl --out ci.csv

Every input is a per_image_scores.jsonl, optionally named as label=path (default label: the
config folder name). All inputs must hold exactly the same images, matched on
(prompt_id, seed); only metrics present for every image of every input are used, and any
other metric is reported on stderr and skipped.

The resampling unit is the prompt: each resample draws prompts with replacement and averages
all their images, so several seeds of one prompt are not treated as independent. For each
input and metric the output gives the mean with a 95% percentile interval; for each input
other than --ref it also gives the mean difference (input - ref) with a paired 95% interval:
every resample draws the same prompts for all inputs, so the shared difficulty of a prompt
cancels out. Means use the per-image scores, which are rounded to 4 decimals, so they can
differ from metrics.csv in the last digit. FID has no per-image score and is not covered.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

METRICS = ("clip_score", "image_reward")
COLUMNS = ["label", "metric", "prompts", "images", "mean", "ci_low", "ci_high",
           "ref", "diff", "diff_low", "diff_high"]


def parse_input(arg):
    """label=path, or a bare path (labelled by its folder); a path may itself contain '='."""
    label, sep, path = arg.partition("=")
    if not sep or not label or any(c in label for c in "/\\") or Path(arg).exists():
        return Path(arg).parent.name, Path(arg)
    return label, Path(path)


def load_scores(path):
    """Return {(prompt_id, seed): {metric: value}}."""
    scores = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["prompt_id"], row["seed"])
            if key in scores:
                raise ValueError(f"{path}: duplicate image {key}")
            scores[key] = {m: float(row[m]) for m in METRICS if m in row}
    if not scores:
        raise ValueError(f"{path}: no images")
    return scores


def align(inputs):
    """Group every input by prompt in one shared order.

    Returns (counts, sums): counts[i] is the number of images of prompt i, and
    sums[label][metric][i] the sum of that metric over those images."""
    ref_label, ref_scores = inputs[0]
    keys = sorted(ref_scores)
    for label, scores in inputs[1:]:
        if set(scores) != set(keys):
            missing, extra = len(set(keys) - set(scores)), len(set(scores) - set(keys))
            raise ValueError(f"{label}: image set differs from {ref_label} "
                             f"({missing} missing, {extra} extra)")
    metrics = [m for m in METRICS if all(m in s[k] for _, s in inputs for k in keys)]
    dropped = sorted({m for _, s in inputs for k in keys for m in s[k]} - set(metrics))
    if dropped:
        print(f"skipping {dropped}: not present for every image of every input", file=sys.stderr)
    if not metrics:
        raise ValueError(f"no metric is present for every image of every input (looked for {METRICS})")

    prompts = sorted({p for p, _ in keys})
    index = {p: i for i, p in enumerate(prompts)}
    rows = np.array([index[p] for p, _ in keys])
    counts = np.bincount(rows, minlength=len(prompts))
    sums = {label: {m: np.bincount(rows, weights=[scores[k][m] for k in keys], minlength=len(prompts))
                    for m in metrics}
            for label, scores in inputs}
    return counts, sums


def bootstrap(counts, sums, ref, n_boot=10000, seed=0, chunk=1000):
    """One shared set of resampled prompts for every input, so differences are paired."""
    n = len(counts)
    rng = np.random.default_rng(seed)
    boot = {label: {m: [] for m in a} for label, a in sums.items()}
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - start), n))
        images = counts[idx].sum(axis=1)
        for label, a in sums.items():
            for m, v in a.items():
                boot[label][m].append(v[idx].sum(axis=1) / images)
    boot = {label: {m: np.concatenate(v) for m, v in a.items()} for label, a in boot.items()}

    rows = []
    for label, a in sums.items():
        for m, v in a.items():
            mean = v.sum() / counts.sum()
            row = {"label": label, "metric": m, "prompts": n, "images": int(counts.sum()), "mean": mean,
                   "ci_low": np.percentile(boot[label][m], 2.5),
                   "ci_high": np.percentile(boot[label][m], 97.5)}
            if label != ref:
                diff = boot[label][m] - boot[ref][m]
                row.update(ref=ref, diff=mean - sums[ref][m].sum() / counts.sum(),
                           diff_low=np.percentile(diff, 2.5), diff_high=np.percentile(diff, 97.5))
            rows.append(row)
    return rows


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="reference input, label=path or path")
    p.add_argument("inputs", nargs="+", help="other inputs, label=path or path")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the table as CSV")
    args = p.parse_args(argv)
    if args.n_boot < 1:
        p.error("--n-boot must be at least 1")

    parsed = [parse_input(a) for a in [args.ref, *args.inputs]]
    labels = [label for label, _ in parsed]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"labels must be unique, got {labels}; name inputs as label=path")
    try:
        counts, sums = align([(label, load_scores(path)) for label, path in parsed])
    except ValueError as e:
        raise SystemExit(str(e))
    rows = bootstrap(counts, sums, labels[0], args.n_boot, args.seed)

    print("\t".join(COLUMNS))
    for row in rows:
        print("\t".join(fmt(row.get(c)) for c in COLUMNS))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
            w.writeheader()
            w.writerows({c: fmt(row.get(c)) for c in COLUMNS} for row in rows)


if __name__ == "__main__":
    main()
