"""Rebuild JQ21's vector figures from the frozen experimental summaries."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as DrawPath
from matplotlib.ticker import LogFormatterMathtext


ROOT = Path(__file__).resolve().parent
WIDTH = 6.35
INK = "#17263A"
SECONDARY = "#49566B"
LINE = "#D5DEE9"
PALE = "#F2F6FA"
BLUE = "#285D91"
TEAL = "#127B78"
RUST = "#AB5635"
PURPLE = "#754980"
SILVER = "#69768A"
DATASETS = {
    "cora": ("Cora", BLUE),
    "ogbn_arxiv_20k": ("ogbn-arxiv 20k", RUST),
    "openalex_20k": ("OpenAlex 20k", TEAL),
    "amazon_computers": ("Amazon Computers", PURPLE),
}
PRIMARY_SEEDS = (191126, 201126)
CONFIRMATION_SEEDS = (119003, 119004)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "mathtext.fontset": "dejavuserif",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": SILVER,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "grid.color": LINE,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.fontsize": 8.2,
    "pdf.fonttype": 42,
})


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def save(fig, name, output, preview):
    fig.savefig(output / f"{name}.pdf", metadata={
        "Creator": "JQ21 plot_figures.py",
        "CreationDate": None,
        "ModDate": None,
    })
    if preview is not None:
        fig.savefig(preview / f"{name}.png", dpi=180)
    plt.close(fig)


def diagram(height):
    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(0, height))
    ax.set_axis_off()
    return fig, ax


def chip(ax, x, y, text, highlighted=False):
    w, h = 0.55, 0.25
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.055",
        facecolor="#DEF2EF" if highlighted else PALE,
        edgecolor=TEAL if highlighted else LINE, linewidth=0.9,
        zorder=4,
    ))
    ax.text(x, y, text, ha="center", va="center", fontsize=10,
            color=TEAL if highlighted else INK, zorder=5)


def setting_figure():
    fig, ax = diagram(2.28)
    titles = ["(a) Keys lost", "(b) One truthful lookup", "(c) Constrained MAP"]
    notes = ["Block membership is known", r"Lookup returns $v_2 \mapsto x_4$",
             "Commit one bijection"]
    y = [1.37, 1.05, 0.73, 0.41]
    for panel, title in enumerate(titles):
        offset = panel * WIDTH / 3
        left, right = offset + 0.43, offset + 1.66
        ax.text(offset + 1.05, 2.07, title, ha="center", weight="bold")
        ax.text(left, 1.68, "Vertex IDs", ha="center", fontsize=8, color=SECONDARY)
        ax.text(right, 1.68, "Feature slots", ha="center", fontsize=8, color=SECONDARY)
        ax.text(offset + 1.05, 0.10, notes[panel], ha="center", fontsize=8.7, color=SECONDARY)
        if panel > 0:
            ax.plot([offset, offset], [0.2, 1.91], color=LINE, lw=0.8)
            links = [(1, 3)] if panel == 1 else [(0, 1), (2, 0), (3, 2), (1, 3)]
            for a, b in links:
                start = (left + 0.27, y[a])
                end = (right - 0.27, y[b])
                route = DrawPath([start, (start[0] + 0.22, start[1]),
                                  (end[0] - 0.22, end[1]), end],
                                 [DrawPath.MOVETO, DrawPath.CURVE4, DrawPath.CURVE4, DrawPath.CURVE4])
                ax.add_patch(FancyArrowPatch(
                    path=route, arrowstyle="-|>", mutation_scale=9,
                    color=TEAL if a == 1 else SILVER, lw=1.8 if a == 1 else 1.05,
                    zorder=3 if a == 1 else 2,
                ))
        for j, level in enumerate(y):
            chip(ax, left, level, rf"$v_{j + 1}$", panel > 0 and j == 1)
            chip(ax, right, level, rf"$x_{j + 1}$", panel > 0 and j == 3)
    return fig


def gain_bound_figure():
    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 2.65))
    fig.subplots_adjust(left=0.092, right=0.971, top=0.814, bottom=0.22, wspace=0.42)
    left.set_title("(a) A transposition attains the bound", loc="left", pad=22)
    left.text(0.5, 1.05, "Only rows 2 and 5 exchange slots", ha="center",
              transform=left.transAxes, fontsize=8.3, color=SECONDARY)
    rows = list(range(1, 7))
    left.bar(rows, [0, 1, 0, 0, 1, 0], width=0.57, color=TEAL, zorder=2)
    left.scatter([1, 3, 4, 6], [0] * 4, marker="_", color=SILVER, s=74,
                 clip_on=False, zorder=3)
    left.axhline(1 / 3, lw=1.0, ls=(0, (4, 3)), color=RUST, zorder=3)
    left.text(3.5, 0.42, "Uniform mean\n$1/3$", ha="center", color=RUST, fontsize=8.5)
    left.set(xlim=(0.5, 6.5), ylim=(-0.03, 1.12), xticks=rows, yticks=[0, 0.5, 1],
             xlabel="Lookup row", ylabel=r"Normalized gain $g_i/g_*$")
    left.grid(axis="y")

    right.set_title("(b) Uniform lookup guarantee", loc="left", pad=22)
    sizes = list(range(2, 25))
    floor = [2 / b for b in sizes]
    right.fill_between(sizes, floor, 1, color="#EAF5F3")
    right.plot(sizes, floor, color=TEAL, lw=2, clip_on=False)
    right.plot(6, 1 / 3, marker="o", ms=5.5, color=RUST, zorder=3)
    right.annotate(r"$2/6=1/3$", (6, 1 / 3), xytext=(9, 0.54), color=RUST, fontsize=10,
                   arrowprops={"arrowstyle": "-", "color": RUST, "lw": 0.8})
    right.text(16.8, 0.245, r"$2/b$", fontsize=11, color=TEAL)
    right.set(xlim=(2, 24), ylim=(0, 1.05), xticks=[2, 6, 12, 18, 24],
              yticks=[0, 0.5, 1], xlabel=r"Block size $b$", ylabel="Fraction of optimal gain")
    right.grid(axis="y")
    return fig


def exchange_figure():
    fig, ax = diagram(2.46)
    ax.text(0.22, 2.24, "(a) Force and complete a row exchange", weight="bold")
    ax.text(3.55, 2.24, "(b) Reassign the cycle's slots", weight="bold")
    ax.plot([3.32, 3.32], [0.15, 2.26], color=LINE, lw=0.9)

    i, k, ell = (0.53, 1.45), (2.80, 1.45), (1.66, 0.61)
    for start, end, color in [(i, k, RUST), (k, ell, TEAL), (ell, i, TEAL)]:
        ax.add_patch(FancyArrowPatch(start, end, color=color, lw=1.9,
                                     arrowstyle="-|>", mutation_scale=13,
                                     shrinkA=12, shrinkB=12))
    for pos, label in [(i, r"$i$"), (k, r"$k$"), (ell, r"$\ell$")]:
        ax.add_patch(Circle(pos, 0.16, edgecolor=LINE, facecolor=PALE, linewidth=0.9, zorder=3))
        ax.text(*pos, label, ha="center", va="center", fontsize=12, zorder=4)
    ax.text(1.67, 1.80, r"Forced arc $i \to k$", color=RUST, ha="center")
    ax.text(1.67, 1.27, r"$C_{ik}$", color=RUST, ha="center", fontsize=11)
    ax.text(1.67, 0.17, r"Shortest return $k \leadsto i$", color=TEAL, ha="center")
    ax.text(0.79, 0.89, r"$C_{\ell i}$", color=TEAL, ha="center", fontsize=10)
    ax.text(2.54, 0.89, r"$C_{k\ell}$", color=TEAL, ha="center", fontsize=10)

    xs = [3.82, 4.62, 5.63]
    for x, label in zip(xs, ["Row", "MAP slot", "New slot"]):
        ax.text(x, 1.78, label, ha="center", weight="bold", fontsize=8.5)
    ax.plot([3.52, 6.13], [1.64, 1.64], color=LINE, lw=0.9)
    for y, row, old, new, accent in [
        (1.41, r"$i$", r"$\sigma(i)$", r"$\sigma(k)$", RUST),
        (1.10, r"$k$", r"$\sigma(k)$", r"$\sigma(\ell)$", TEAL),
        (0.79, r"$\ell$", r"$\sigma(\ell)$", r"$\sigma(i)$", TEAL),
    ]:
        for x, label, color in zip(xs, [row, old, new], [INK, INK, accent]):
            ax.text(x, y, label, ha="center", va="center", color=color, fontsize=11)
    ax.plot([3.52, 6.13], [0.59, 0.59], color=LINE, lw=0.9)
    ax.text(4.83, 0.24, r"Forced loss $D_{i,\sigma(k)}=C_{ik}+d(k,i)$",
            ha="center", va="center", fontsize=10.8)
    return fig


def split_legend(seeds):
    return [Line2D([], [], color=INK, marker=marker, mfc=fill, lw=0,
                   ms=5.1, label=f"Split {seed}")
            for seed, marker, fill in zip(seeds, ("o", "s"), (INK, "white"))]


def dataset_axis(ax, labels=True):
    ax.set(yticks=range(4), yticklabels=[label for label, _ in DATASETS.values()], ylim=(3.54, -0.55))
    ax.tick_params(axis="y", length=0, labelleft=labels, pad=6)
    ax.spines["left"].set_visible(False)
    for row in (0.5, 1.5, 2.5):
        ax.axhline(row, color=LINE, lw=0.6, zorder=0)
    ax.grid(axis="x", linestyle=":")


def paired_forest(records, seeds, fields, limits, ticks, x_label, threshold=None):
    fig, ax = plt.subplots(figsize=(WIDTH, 2.66))
    fig.subplots_adjust(left=0.233, right=0.973, top=0.825, bottom=0.245)
    dataset_axis(ax)
    ax.set(xlim=limits, xticks=ticks, xlabel=x_label)
    ax.axvline(0, color=SILVER, lw=1.0, zorder=1)
    handles = split_legend(seeds)
    if threshold is not None:
        ax.axvline(threshold, color=RUST, ls=(0, (4, 3)), lw=1.05, zorder=1)
        handles.append(Line2D([], [], color=RUST, ls=(0, (4, 3)), lw=1.05,
                              label="Prespecified 5-point gate"))
    ax.legend(handles=handles, ncol=len(handles), loc="lower left", bbox_to_anchor=(0, 1.01),
              borderaxespad=0, handletextpad=0.45, columnspacing=1.3)
    by_cell = {(r["dataset"], int(r["seed"])): r for r in records}
    for y, (dataset, (_, color)) in enumerate(DATASETS.items()):
        for split, seed in enumerate(seeds):
            record = by_cell[(dataset, seed)]
            v, low, high = (100 * float(record[key]) for key in fields)
            ax.errorbar(v, y + (split - 0.5) * 0.27,
                        xerr=[[v - low], [high - v]], capsize=2.4, capthick=1,
                        elinewidth=1.25, markersize=5.3,
                        fmt="o" if split == 0 else "s", color=color,
                        markerfacecolor=color if split == 0 else "white", zorder=3)
    return fig


def regime_figure(records):
    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 2.65), sharey=True)
    fig.subplots_adjust(left=0.226, right=0.968, top=0.735, bottom=0.218, wspace=0.18)
    left.set_title("(a) Aligned contexts", loc="left", pad=4)
    right.set_title("(b) Shuffled contexts", loc="left", pad=4)
    dataset_axis(left)
    dataset_axis(right, labels=False)
    left.set(xlim=(0, 21), xticks=[0, 5, 10, 15, 20], xlabel="Gain over random (pp)")
    right.set(xlim=(-0.11, 1.72), xticks=[0, 0.5, 1, 1.5],
              xlabel="Exact recovery (%)")
    right.axvline(100 / 120, color=SILVER, ls=(0, (3, 3)), lw=1)

    by_cell = {(r["dataset"], int(r["seed"]), r["regime"], r["control"]): r for r in records}
    for y, dataset in enumerate(DATASETS):
        for group, (regime, color) in enumerate([("uniform", BLUE), ("banded", RUST)]):
            for split, seed in enumerate(PRIMARY_SEEDS):
                offset = y + (group - 0.5) * 0.36 + (split - 0.5) * 0.17
                gain = by_cell[(dataset, seed, regime, "aligned")]["vs_random"]
                recovery = by_cell[(dataset, seed, regime, "shuffled")]["value"]
                for ax, value in [(left, gain), (right, recovery)]:
                    ax.plot(100 * float(value), offset, color=color, markersize=4.6,
                            marker="o" if split == 0 else "s", linestyle="none",
                            mfc=color if split == 0 else "white", zorder=3)
    handles = [Line2D([], [], color=color, lw=2.5, label=label)
               for color, label in [(BLUE, "Uniform"), (RUST, "Context-banded")]]
    handles += split_legend(PRIMARY_SEEDS)
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.972, 0.98),
               ncols=4, handlelength=1.1, columnspacing=1.2)
    return fig


def block_size_axes():
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.70), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.108, right=0.975, top=0.813, bottom=0.151, hspace=0.44, wspace=0.15)
    for ax, (label, _) in zip(axes.flat, DATASETS.values()):
        ax.set_title(label, loc="left", pad=3)
        ax.set(xlim=(-0.15, 3.23), ylim=(-5, 105), xticks=range(4), xticklabels=[4, 8, 12, 24],
               yticks=[0, 25, 50, 75, 100])
        ax.grid(axis="y")
    fig.supylabel("Exact recovery (%)", x=0.014, y=0.5, fontsize=9)
    fig.supxlabel("Records per block", x=0.54, y=0.033, fontsize=9)
    return fig, axes


def block_sizes_figure(records):
    fig, axes = block_size_axes()
    measurements = {(r["dataset"], int(r["size"])): r for r in records if r["fit"] == "archived_b6"}
    policies = [
        ("value", "Whole-block value", TEAL, "o", "-", (4, 8, 12, 24)),
        ("exact_entropy", "Exact entropy", RUST, "^", (0, (1, 2)), (4, 8, 12)),
        ("random", "Uniform row", SILVER, "s", (0, (3, 2)), (4, 8, 12, 24)),
    ]
    for ax, dataset in zip(axes.flat, DATASETS):
        for key, name, color, marker, style, sizes in policies:
            rates = [100 * float(measurements[dataset, b][key]) for b in sizes]
            ax.plot(range(len(sizes)), rates, lw=1.45, marker=marker, markersize=4.2,
                    linestyle=style, mfc=color if key == "exact_entropy" else "white",
                    color=color, label=name)
        rate = 100 * float(measurements[dataset, 24]["sinkhorn_entropy"])
        ax.plot(3.11, rate, marker="^", color=RUST, ms=5.3, mfc="white",
                linestyle="none", label="Sinkhorn, 24 only")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.992),
               ncols=4, columnspacing=1.05, handlelength=1.45)
    return fig


def size_calibration_figure(records):
    fig, axes = block_size_axes()
    measurements = {(r["dataset"], int(r["size"]), r["fit"]): 100 * float(r["value"])
                    for r in records}
    for ax, dataset in zip(axes.flat, DATASETS):
        archived = [measurements[dataset, b, "archived_b6"] for b in (4, 8, 12, 24)]
        native = [measurements[dataset, b, "native_gibbs"] for b in (4, 8, 12)]
        ax.plot(range(4), archived, lw=1.45, color=BLUE, ls=(0, (3, 2)), ms=6, marker="o",
                mfc="none", zorder=3, label="Six-row fit")
        ax.plot(range(3), native, lw=1.45, color=TEAL, ms=3.5, marker="s", mfc=TEAL,
                label="Native-size Gibbs fit (4--12)")
        for x, key, name, marker in [(2.88, "native_pair_24", "Pair-conditional fit (24)", "D"),
                                     (3.12, "native_halves_24", "Half-conditional fit (24)", "x")]:
            ax.plot(x, measurements[dataset, 24, key], color=RUST, ms=4.4, marker=marker,
                    linestyle="none", label=name)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, [label.replace("--", "–") for label in labels], loc="upper center",
               bbox_to_anchor=(0.55, 0.994), ncols=2, columnspacing=1.7, handlelength=1.45)
    return fig


def kernel_timing_figure(aggregate):
    fig, ax = plt.subplots(figsize=(WIDTH, 2.98))
    fig.subplots_adjust(left=0.142, right=0.969, top=0.835, bottom=0.215)
    methods = [
        ("exchange_exact", "Exchange paths", TEAL, "o"),
        ("forced_hungarian_exact", "Repeated forced assignments", RUST, "s"),
    ]
    for key, label, color, marker in methods:
        entries = [(r["n"], 1000 * r["median_seconds"][key])
                   for r in aggregate if key in r["median_seconds"]]
        sizes, medians = zip(*entries)
        ax.plot(sizes, medians, marker=marker, color=color, ms=4.8, lw=1.6, label=label)
    ax.set(xscale="log", yscale="log", xlim=(3.65, 282), ylim=(0.008, 6500),
           xlabel="Records per block", ylabel="Median kernel time (ms)")
    sizes = [r["n"] for r in aggregate]
    ax.set_xticks(sizes, labels=sizes)
    ax.set_xticks([], minor=True)
    ax.set_yticks([0.01, 0.1, 1, 10, 100, 1000])
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.grid(axis="y", which="major")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncols=2,
              borderaxespad=0, columnspacing=1.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=ROOT / "figure_data",
                        help="Directory containing the frozen experiment summaries")
    parser.add_argument("--stress", type=Path, default=ROOT / "revision_data" / "block_stress" / "summary.csv",
                        help="Summary of the paired block-size calibration replay")
    parser.add_argument("--output", type=Path, default=ROOT / "figures",
                        help="Output directory for the vector PDFs")
    parser.add_argument("--preview-dir", type=Path, help="Optional directory for PNG previews")
    args = parser.parse_args()

    rates = read_csv(args.experiments / "rates.csv")
    block_sizes = read_csv(args.stress)
    confirmation = read_csv(args.experiments / "b12_confirmation" / "accelerated_results" / "paired_summary.csv")
    timing = json.loads((args.experiments / "sensitivity_results" / "summary.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)

    aligned = [r for r in rates if (r["regime"], r["control"]) == ("uniform", "aligned")]
    exact_entropy = [r for r in confirmation if (r["policy"], r["comparator"]) == ("value", "exact_entropy")]
    figures = {
        "setting": setting_figure(),
        "gain_bound": gain_bound_figure(),
        "exchange_paths": exchange_figure(),
        "primary_gain": paired_forest(aligned, PRIMARY_SEEDS, ("vs_random", "vs_random_lo", "vs_random_hi"),
                                      (-0.60, 21.3), [0, 5, 10, 15, 20],
                                      "Whole-block value minus uniform row (percentage points)", threshold=5),
        "regime_controls": regime_figure(rates),
        "block_sizes": block_sizes_figure(block_sizes),
        "size_calibration": size_calibration_figure(block_sizes),
        "exact_entropy_confirmation": paired_forest(exact_entropy, CONFIRMATION_SEEDS, ("difference", "ci_low", "ci_high"),
                                                     (-8.2, 10.1), [-8, -4, 0, 4, 8],
                                                     "Whole-block value minus exact entropy (percentage points)"),
        "kernel_timing": kernel_timing_figure(timing["timing_aggregate_across_four_score_families"]),
    }
    for name, fig in figures.items():
        save(fig, name, args.output, args.preview_dir)


if __name__ == "__main__":
    main()
