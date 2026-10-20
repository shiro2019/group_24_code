"""Paired bootstrap confidence intervals for the per-image metrics of src.evaluate.

    python -m src.bootstrap_ci --ref base50=outputs/e1/eval/sdxl_base/steps50_cfg7.5/per_image_scores.jsonl \
        turbo4=<...>/per_image_scores.jsonl base8=<...>/per_image_scores.jsonl --out ci.csv

Every input is a per_image_scores.jsonl, optionally named as label=path (default label: the
config folder name). All inputs must hold exactly the same images, matched on
(prompt_id, seed). For each input and metric the output gives the mean with a 95%
percentile interval; for each input other than --ref it also gives the mean difference
(input - ref) with a paired 95% interval: every resample draws the same prompts for both
inputs, so the shared difficulty of a prompt cancels out. FID has no per-image score and
is not covered.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = ("clip_score", "image_reward")
COLUMNS = ["label", "metric", "n", "mean", "ci_low", "ci_high",
           "ref", "diff", "diff_low", "diff_high"]


def parse_input(arg):
    label, sep, path = arg.partition("=")
    if not sep:
        path = arg
        label = Path(arg).parent.name
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
    return scores


def align(inputs):
    """Stack every input into {label: {metric: array}} in one shared image order."""
    keys = sorted(inputs[0][1])
    for label, scores in inputs[1:]:
        if set(scores) != set(keys):
            missing, extra = len(set(keys) - set(scores)), len(set(scores) - set(keys))
            raise ValueError(f"{label}: image set differs from {inputs[0][0]} "
                             f"({missing} missing, {extra} extra)")
    return {label: {m: np.array([scores[k][m] for k in keys])
                    for m in METRICS if all(m in scores[k] for k in keys)}
            for label, scores in inputs}


def resample_means(values, idx):
    return values[idx].mean(axis=1)


def bootstrap(arrays, ref, n_boot=10000, seed=0, chunk=1000):
    """One shared set of resampled indices for every input, so differences are paired."""
    n = len(next(iter(arrays[ref].values())))
    rng = np.random.default_rng(seed)
    means = {label: {m: [] for m in a} for label, a in arrays.items()}
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - start), n))
        for label, a in arrays.items():
            for m, v in a.items():
                means[label][m].append(resample_means(v, idx))
    means = {label: {m: np.concatenate(v) for m, v in a.items()} for label, a in means.items()}

    rows = []
    for label, a in arrays.items():
        for m, v in a.items():
            boot = means[label][m]
            row = {"label": label, "metric": m, "n": n, "mean": v.mean(),
                   "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)}
            if label != ref and m in arrays[ref]:
                diff = boot - means[ref][m]
                row.update(ref=ref, diff=v.mean() - arrays[ref][m].mean(),
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

    parsed = [parse_input(a) for a in [args.ref, *args.inputs]]
    labels = [label for label, _ in parsed]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"labels must be unique, got {labels}; name inputs as label=path")
    arrays = align([(label, load_scores(path)) for label, path in parsed])
    rows = bootstrap(arrays, parsed[0][0], args.n_boot, args.seed)

    print("\t".join(COLUMNS))
    for row in rows:
        print("\t".join(fmt(row.get(c)) for c in COLUMNS))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows({c: fmt(row.get(c)) for c in COLUMNS} for row in rows)


if __name__ == "__main__":
    main()
