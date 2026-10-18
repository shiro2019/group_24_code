"""Report figures from the summary tables (D).

    python -m src.plot_results --tables outputs/tables --out outputs/figures

Reads the six tables A builds from the merged metrics CSV (see the tables' README.md)
and draws one figure per table, as PNG for slides and PDF for the report:

    e1_quality_latency   e1_main.csv           RQ1: every method's frozen points
    e2_sweeps            e2_sweep.csv          RQ2: tuning sweeps, selected points marked
    e3_adapter_scale     e3_scale.csv          RQ2: LCM-LoRA adapter scale
    e4_hybrid            e1_main.csv           RQ3: hybrid against the single methods
    e4_strength          e4_strength_tune.csv  RQ3: refine strength on the tuning split
    diversity            diversity.csv         seed diversity of the quality points

latency_sorted.csv holds the same rows as e1_main.csv and needs no figure of its own.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Reference categorical palette (dataviz skill, light mode), used in this fixed order.
# Each method also gets its own marker, and lines are labelled directly, so identity
# never rests on colour alone.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
CONTEXT = "#b4b2ab"  # recessive grey for context points
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

METHODS = ["SDXL + DPM-Solver++", "LCM-LoRA", "SDXL-Turbo", "SDXL-Lightning"]
SHORT = {"SDXL + DPM-Solver++": "SDXL base", "LCM-LoRA": "LCM-LoRA",
         "SDXL-Turbo": "Turbo", "SDXL-Lightning": "Lightning"}
HYBRID = "E4 Turbo draft + SDXL refine"
METRICS = [("image_reward", "ImageReward", "higher is better"),
           ("clip_score", "CLIP Score", "higher is better"),
           ("fid", "FID", "lower is better")]
LATENCY_LABEL = "End-to-end latency, s (L40S, log scale)"


def style(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_2, labelsize=9)


def title(ax, name, hint=""):
    ax.set_title(f"{name}  ({hint})" if hint else name, color=TEXT, fontsize=10, loc="left")


def log_latency_axis(ax, ticks=(0.1, 0.2, 0.5, 1, 2, 5)):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel(LATENCY_LABEL, color=TEXT_2, fontsize=9)


# Offsets (points) tried in order for each label; the first one that overlaps neither an
# earlier label nor a data point wins.
OFFSETS = [(8, 5, "left", "bottom"), (8, -5, "left", "top"), (-8, 5, "right", "bottom"),
           (-8, -5, "right", "top"), (0, 10, "center", "bottom"), (0, -10, "center", "top"),
           (14, 0, "left", "center"), (-14, 0, "right", "center")]


def obstacles(ax):
    """Display-space boxes around every data point and along every line segment."""
    boxes = []
    for coll in ax.collections:
        for x, y in coll.get_offsets():
            px, py = ax.transData.transform((x, y))
            boxes.append((px - 6, py - 6, px + 6, py + 6))
    for line in ax.lines:
        pts = ax.transData.transform(line.get_xydata())
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            for t in [i / 20 for i in range(21)]:
                px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                boxes.append((px - 3, py - 3, px + 3, py + 3))
        for px, py in pts:
            boxes.append((px - 6, py - 6, px + 6, py + 6))
    return boxes


def place_labels(ax, items, color=TEXT_2, taken=None):
    """items: [(text, x, y)]. Greedy placement that avoids labels, points and lines.
    Pass the returned list back in to keep later labels clear of these ones."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    taken = obstacles(ax) if taken is None else taken
    for text, x, y in items:
        best = None
        for dx, dy, ha, va in OFFSETS:
            a = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=8, color=color)
            b = a.get_window_extent(renderer)
            box = (b.x0 - 1, b.y0 - 1, b.x1 + 1, b.y1 + 1)
            clash = any(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3] for t in taken)
            if not clash:
                best = box
                break
            a.remove()
        if best is None:  # nothing free: keep the first position
            a = ax.annotate(text, (x, y), xytext=OFFSETS[0][:2], textcoords="offset points",
                            ha=OFFSETS[0][2], va=OFFSETS[0][3], fontsize=8, color=color)
            b = a.get_window_extent(renderer)
            best = (b.x0, b.y0, b.x1, b.y1)
        taken.append(best)
    return taken


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def method_line(ax, g, metric, i, label=None):
    g = g.sort_values("latency_s")
    ax.plot(g["latency_s"], g[metric], color=SERIES[i], marker=MARKERS[i], linewidth=2,
            markersize=8, markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3)


def e1_figure(e1, out):
    """RQ1: quality-priority and lowest-step points of each method, joined by a line."""
    main = e1[e1["method"].isin(METHODS) & e1["role"].isin(["quality", "lowest-step"])]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), constrained_layout=True)
    for ax, (metric, name, hint) in zip(axes, METRICS):
        style(ax)
        items = []
        for i, m in enumerate(METHODS):
            g = main[main["method"] == m]
            method_line(ax, g, metric, i, label=m)
            items += [(f"{SHORT[m]} {r['steps']}", r["latency_s"], r[metric]) for _, r in g.iterrows()]
        log_latency_axis(ax)
        ax.margins(x=0.12, y=0.12)
        title(ax, name, hint)
        place_labels(ax, items)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("E1: frozen configurations on the eval split (2,000 COCO prompts); "
                 "each line joins a method's lowest-step and quality-priority points",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    save(fig, out, "e1_quality_latency")


def e2_figure(e2, out):
    """RQ2: ImageReward and CLIP against steps for every tuning sweep."""
    order = ["sdxl_base", "lcm_lora", "sdxl_turbo", "sdxl_lightning"]
    names = dict(zip(order, METHODS))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2), constrained_layout=True, sharex="col")
    for col, m in enumerate(order):
        g = e2[(e2["method"] == m) & (e2["selected"] != "exploratory")]
        cfgs = sorted(g["cfg"].unique())
        for row, (metric, name, _) in enumerate(METRICS[:2]):
            ax = axes[row, col]
            style(ax)
            selected = []
            for i, c in enumerate(cfgs):
                s = g[g["cfg"] == c].sort_values("steps")
                ax.plot(s["steps"], s[metric], color=SERIES[i], marker=MARKERS[i], linewidth=2,
                        markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                        label=f"CFG {c:g}" if len(cfgs) > 1 else None, zorder=3)
                for _, r in s[s["selected"].isin(["quality", "lowest-step"])].iterrows():
                    ax.scatter(r["steps"], r[metric], s=170, facecolors="none", edgecolors=TEXT,
                               linewidths=1.2, zorder=4)
                    selected.append((r["selected"], r["steps"], r[metric]))
            ax.set_xscale("log")
            ax.set_xticks(sorted(g["steps"].unique()))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
            ax.margins(x=0.1, y=0.18)
            place_labels(ax, selected)
            if row == 0:
                title(ax, names[m])
                if len(cfgs) > 1:
                    ax.legend(frameon=False, fontsize=8, loc="lower right")
            else:
                ax.set_xlabel("Steps (log scale)" + ("\none checkpoint per step count"
                                                     if m == "sdxl_lightning" else ""),
                              color=TEXT_2, fontsize=9)
            if col == 0:
                ax.set_ylabel(name, color=TEXT, fontsize=10)
    fig.suptitle("E2: tuning sweeps (500 COCO prompts); circled points are the frozen configurations",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    save(fig, out, "e2_sweeps")


def e3_figure(e3, out):
    """RQ2: LCM-LoRA adapter scale; tuning sweep with the eval re-tests."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, (metric, name, hint), ev in zip(axes, METRICS[:2], ("eval_ir", "eval_clip")):
        style(ax)
        tune = {"image_reward": "tune_ir", "clip_score": "tune_clip"}[metric]
        ax.plot(e3["lora_scale"], e3[tune], color=SERIES[0], marker="o", linewidth=2, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5, label="tuning split (500)", zorder=3)
        ev_rows = e3.dropna(subset=[ev])
        ax.scatter(ev_rows["lora_scale"], ev_rows[ev], s=70, color=SERIES[1], marker="s",
                   edgecolors="white", linewidths=1.5, label="eval split re-test (2,000)", zorder=4)
        ax.set_xticks(e3["lora_scale"])
        ax.set_xlabel("Adapter scale (LoRA not fused)", color=TEXT_2, fontsize=9)
        ax.margins(y=0.15)
        title(ax, name, hint)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("E3: LCM-LoRA adapter scale (8 steps, guidance 2.0)", color=TEXT, fontsize=11,
                 x=0.01, ha="left")
    save(fig, out, "e3_adapter_scale")


def e4_figure(e1, out):
    """RQ3: the hybrid and its no-refine control against every single-method point."""
    single = e1[e1["method"].isin(METHODS)]
    hyb = e1[e1["method"] == HYBRID]
    refined = hyb[hyb["config"] != "draft_only"]
    draft = hyb[hyb["config"] == "draft_only"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), constrained_layout=True)
    for ax, (metric, name, hint) in zip(axes, METRICS):
        style(ax)
        ax.scatter(single["latency_s"], single[metric], s=45, color=CONTEXT, zorder=2,
                   label="single methods (E1 points, grey)")
        for g, i, label in ((refined, 1, "Hybrid: Turbo 1-step draft + SDXL refine, strength 0.5"),
                            (draft, 2, "Turbo 1-step draft only, upscaled (no refine; ~ Turbo 1 step)")):
            ax.scatter(g["latency_s"], g[metric], s=90, color=SERIES[i], marker=MARKERS[i],
                       edgecolors="white", linewidths=1.5, label=label, zorder=4)
        log_latency_axis(ax)
        ax.margins(x=0.12, y=0.12)
        title(ax, name, hint)
        taken = place_labels(ax, [("hybrid", r["latency_s"], r[metric]) for _, r in refined.iterrows()]
                             + [("draft only", r["latency_s"], r[metric]) for _, r in draft.iterrows()],
                             color=TEXT)
        # Turbo 1 step is the same model as the draft and sits under it; its label would only
        # crowd the draft's (the legend says so).
        others = single[~((single["method"] == "SDXL-Turbo") & (single["steps"].astype(str) == "1"))]
        place_labels(ax, [(f"{SHORT[r['method']]} {r['steps']}", r["latency_s"], r[metric])
                          for _, r in others.sort_values("latency_s").iterrows()], taken=taken)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("E4: two-stage hybrid against single few-step methods on the eval split "
                 "(2,000 COCO prompts)", color=TEXT, fontsize=11, x=0.01, ha="left")
    save(fig, out, "e4_hybrid")


def e4_strength_figure(e4, out):
    """RQ3: refine strength on the tuning split, the no-refine control as a reference."""
    draft = e4[e4["config"] == "draft_only"].iloc[0]
    ref = e4[e4["config"] != "draft_only"].copy()
    ref["strength"] = ref["strength"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, (metric, name, hint) in zip(axes, METRICS[:2]):
        style(ax)
        ax.plot(ref["strength"], ref[metric], color=SERIES[1], marker="s", linewidth=2, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        ax.set_xticks(ref["strength"])
        ax.set_xlabel("Refine strength (labels: latency on L40S)", color=TEXT_2, fontsize=9)
        ax.margins(x=0.1, y=0.2)
        title(ax, name, hint)
        # A data-space line rather than axhline so place_labels sees it at the right x.
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [draft[metric]] * 2, color=SERIES[2], linewidth=1.5, linestyle="--", zorder=2)
        ax.set_xlim(lo, hi)
        taken = place_labels(ax, [("no refine (draft only)", ref["strength"].min(), draft[metric])])
        place_labels(ax, [(f"{r['latency_s']:.2f} s", r["strength"], r[metric])
                          for _, r in ref.iterrows()], taken=taken)
    fig.suptitle("E4: refine strength on the tuning split (500 COCO prompts, nominal 10 refine steps)",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    save(fig, out, "e4_strength")


def diversity_figure(div, out):
    """Seed diversity next to the quality of the same images. A dot plot, because the
    differences are small and bars would have to start at zero to be honest."""
    div = div.sort_values("lpips_diversity")
    labels = [f"{SHORT[m]} ({c})" for m, c in zip(div["method"], div["config"])]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 2.9), constrained_layout=True, sharey=True)
    for ax, (col, name, hint) in zip(axes, [("lpips_diversity", "Mean pairwise LPIPS", "seed diversity"),
                                            ("image_reward", "ImageReward", "higher is better"),
                                            ("clip_score", "CLIP Score", "higher is better")]):
        style(ax)
        ax.grid(axis="y", visible=False)
        ys = range(len(div))
        ax.hlines(ys, div[col].min(), div[col], color=GRID, linewidth=1.5, zorder=2)
        ax.scatter(div[col], ys, s=70, color=SERIES[0], edgecolors="white", linewidths=1.5, zorder=3)
        for y, v in zip(ys, div[col]):
            ax.annotate(f"{v:.3f}" if v < 10 else f"{v:.2f}", (v, y), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8, color=TEXT_2)
        ax.set_yticks(list(ys), labels)
        ax.margins(x=0.15, y=0.2)
        title(ax, name, hint)
    fig.suptitle("Diversity: 100 prompts x 8 seeds per method (quality-priority points)",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    save(fig, out, "diversity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True, help="folder with the summary tables")
    ap.add_argument("--out", default="outputs/figures")
    args = ap.parse_args()
    tables, out = Path(args.tables), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(tables / name)
    e1 = read("e1_main.csv")
    print(f"figures in {out}:")
    e1_figure(e1, out)
    e2_figure(read("e2_sweep.csv"), out)
    e3_figure(read("e3_scale.csv"), out)
    e4_figure(e1, out)
    e4_strength_figure(read("e4_strength_tune.csv"), out)
    diversity_figure(read("diversity.csv"), out)


if __name__ == "__main__":
    main()
