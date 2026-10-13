"""Merge everyone's metrics CSVs and draw the report figures (D).

    python -m src.plot_results --csv metrics_A.csv metrics_D.csv --out figures

Each member's outputs/metrics/metrics.csv is merged; for every
(experiment, split, method, config, metric) the newest row by timestamp wins, as the
README asks. Latency comes from src.timing / src.hybrid --timing and is matched to a
quality row by (method, config) whatever split it was timed on, since timing does not
depend on the prompts. Writes merged.csv, a wide table of the eval-split results and
the figures.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

KEY = ["experiment", "split", "method", "config", "metric"]
QUALITY = ("fid", "clip_score", "image_reward")
LATENCY = ("latency_median_s", "latency_iqr_s", "peak_mem_gb", "unet_calls_per_image")

# Reference categorical palette (dataviz skill, light mode), slots in fixed order.
# Scatter-like charts use at most the first three slots.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PANEL = {"image_reward": ("ImageReward", "higher is better"),
         "clip_score": ("CLIP Score", "higher is better"),
         "fid": ("FID", "lower is better")}


def merge(paths):
    df = pd.concat([pd.read_csv(p, dtype=str) for p in paths], ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates(KEY, keep="last")
    return df.sort_values(KEY).reset_index(drop=True)


def eval_table(df):
    """One row per eval-split config: quality metrics plus its measured latency."""
    q = df[(df["split"] == "eval") & df["metric"].isin(QUALITY)]
    wide = q.pivot_table(index=["experiment", "method", "config"], columns="metric",
                         values="value", aggfunc="first").reset_index()
    t = df[df["metric"].isin(LATENCY)].sort_values("timestamp")
    t = t.drop_duplicates(["method", "config", "metric"], keep="last")
    lat = t.pivot_table(index=["method", "config"], columns="metric", values="value",
                        aggfunc="first").reset_index()
    out = wide.merge(lat, on=["method", "config"], how="left")
    num = [c for c in (*QUALITY, *LATENCY) if c in out]
    out[num] = out[num].apply(pd.to_numeric)
    return out


def _style(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_2, labelsize=9)


def _label_line(ax, g, metric, labels):
    """Label each point of a line on the side away from the line: below-right when the
    next segment rises, above-right when it falls; the last point is labelled below."""
    ys = list(g[metric])
    for i, (x, y, text) in enumerate(zip(g["latency_median_s"], ys, labels)):
        if i == len(ys) - 1:
            xy, ha, va = (0, -10), "center", "top"
        elif ys[i + 1] > y:
            xy, ha, va = (8, -6), "left", "top"
        else:
            xy, ha, va = (8, 6), "left", "bottom"
        ax.annotate(text, (x, y), xytext=xy, textcoords="offset points", ha=ha, va=va,
                    fontsize=8, color=TEXT_2)


def e4_figure(table, out):
    """RQ3: E4 hybrid and its no-refine control against the baseline steps (eval split)."""
    base = table[(table["method"] == "sdxl_base") & table["latency_median_s"].notna()].copy()
    base["steps"] = base["unet_calls_per_image"].astype(int)
    base = base.sort_values("latency_median_s")
    hyb = table[table["method"] == "hybrid"]
    refined = hyb[hyb["config"] != "draft_only"]
    draft = hyb[hyb["config"] == "draft_only"]
    groups = [("SDXL baseline (4 / 8 / 50 steps)", base, SERIES[0]),
              ("Hybrid: Turbo draft + SDXL refine, strength 0.5", refined, SERIES[1]),
              ("Turbo draft only (no refine)", draft, SERIES[2])]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, metric in zip(axes, PANEL):
        _style(ax)
        for label, g, color in groups:
            if g.empty or metric not in g:
                continue
            ax.plot(g["latency_median_s"], g[metric], color=color, linewidth=2,
                    marker="o", markersize=8, markeredgecolor="white", markeredgewidth=2,
                    label=label, zorder=3)
        _label_line(ax, base, metric, [f"{s} steps" for s in base["steps"]])
        # The hybrid sits just left of the 8-step point; the draft is next to the y axis.
        for g, text, dx, ha in ((refined, "strength 0.5", -9, "right"), (draft, "draft only", 9, "left")):
            for _, r in g.iterrows():
                ax.annotate(text, (r["latency_median_s"], r[metric]), xytext=(dx, 6),
                            textcoords="offset points", ha=ha, fontsize=8, color=TEXT_2)
        ax.margins(x=0.08, y=0.14)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.3, 1, 3, 6])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("End-to-end latency, s (L40S, log scale)", color=TEXT_2, fontsize=9)
        name, hint = PANEL[metric]
        ax.set_title(f"{name}  ({hint})", color=TEXT, fontsize=10, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=TEXT)
    fig.suptitle("E4 (RQ3): quality vs latency on the eval split, 2,000 COCO prompts",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"e4_quality_latency.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="every member's metrics.csv")
    ap.add_argument("--out", default="outputs/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = merge(args.csv)
    df.to_csv(out / "merged.csv", index=False)
    table = eval_table(df)
    table.to_csv(out / "eval_table.csv", index=False)
    print(table.to_string(index=False))
    e4_figure(table, out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
