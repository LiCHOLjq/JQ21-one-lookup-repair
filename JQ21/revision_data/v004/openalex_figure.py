"""Draw the matched OpenAlex ablation using the archived evaluation rates."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    here = Path(__file__).resolve().parent
    records = {}
    for view in ("matched_full", "title_abstract"):
        with (here / f"openalex_{view}.csv").open(newline="") as source:
            for row in csv.DictReader(source):
                records[view, row["edges"], row["seed"]] = row
    groups = [("matched_full", "all"), ("title_abstract", "all"),
              ("matched_full", "citation_author"), ("title_abstract", "citation_author")]
    labels = ["Original graph, text + metadata", "Original graph, title + abstract",
              "Restricted graph, text + metadata", "Restricted graph, title + abstract"]
    styles = [("unqueried", "No lookup", "o", "#586574"),
              ("random", "Uniform lookup", "D", "#D27124"),
              ("value", "Whole-block value", "s", "#057A79")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.50), sharey=True, sharex=True)
    for ax, seed in zip(axes, ("191126", "201126")):
        for i, (view, graph) in enumerate(groups):
            row = records[view, graph, seed]
            values = [100 * float(row[key]) for key, *_ in styles]
            ax.plot([min(values), max(values)], [i, i], color="#ADB7BB", lw=1.3, zorder=1)
            for x, (_, _, mark, color) in zip(values, styles):
                ax.scatter(x, i, color=color, marker=mark, s=27, zorder=3)
        ax.set_title(f"Split {seed}", fontsize=10)
        ax.set(xlim=(60, 101.5), xticks=[60, 70, 80, 90, 100], ylim=(3.4, -.4))
        ax.set_yticks(range(4), labels)
        ax.xaxis.grid(True, color="#E3E8E9", lw=.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=8, pad=3)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    handles = [Line2D([], [], marker=m, lw=0, color=c, markersize=5, label=l)
               for _, l, m, c in styles]
    fig.subplots_adjust(left=.30, right=.985, top=.84, bottom=.30, wspace=.15)
    fig.text(.64, .13, "Exact recovery (%)", ha="center", fontsize=9)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    out = here.parents[1] / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "openalex_ablation.pdf", bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
