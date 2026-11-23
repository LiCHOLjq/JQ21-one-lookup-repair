"""Render the manuscript's policy tables from the saved revision CSV files."""

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATASETS = (
    ("cora", "Cora"),
    ("ogbn_arxiv_20k", r"\texttt{ogbn-arxiv}"),
    ("openalex_20k", "OpenAlex"),
    ("amazon_computers", r"\shortstack[l]{Amazon\\Computers}"),
)
POLICIES = (
    ("no_lookup", "No lookup"),
    ("random", "Uniform row"),
    ("raw_gap", "Raw-score gap"),
    ("exact_entropy", "Exact entropy"),
    ("value", "Whole-block value"),
    ("expected_rows", "Expected rows"),
    ("least_confident", "Least confidence"),
    ("forced_gap", "Forced gap"),
    ("sinkhorn_entropy", "Sinkhorn entropy"),
    ("hindsight", "Hindsight ceiling"),
)
STAGES = (
    ("six_row", 6, ("191126", "201126"), "action_tie_average"),
    ("twelve_row", 12, ("119003", "119004"), "action_and_decoder_tie_average"),
)


def read_csv(name):
    with (HERE / name).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_tex(name, lines):
    (HERE.parent / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def pct(value):
    value = 100 * float(value)
    return f"{0 if abs(value) < .005 else value:.2f}"


def delta(value):
    number = pct(value)
    return "+" + number if float(number) > 0 else number


def interval(row):
    return f"[{pct(row['ci_low'])},{pct(row['ci_high'])}]"


def tex_row(*cells):
    return " & ".join(cells) + r" \\"


def indexed(rows):
    return {
        (r["stage"], r["dataset"], r["seed"], r["convention"], r["endpoint"], r["policy"]): r
        for r in rows if r["condition"] == "aligned" and r["regime"] == "uniform"
    }


def primary_results(index, diagnostic):
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Exact recovery on aligned, uniformly formed six-row blocks under the original first-maximum action convention. Rates are percentages. Contrasts $\Delta_{\mathrm U}$ and $\Delta_{\mathrm E}$ are whole-block value minus uniform lookup and exact marginal entropy, respectively, in percentage points with paired 95\% intervals. Contrasts use unrounded per-block outcomes.}",
        r"\label{tab:primary-six}",
        r"\ifdefined\JournalReviewCopy\fontsize{9}{11}\selectfont\else\footnotesize\fi",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\begin{tabular}{@{}lrrrrrrrll@{}}",
        r"\toprule",
        tex_row("Graph", "Split", "$n$", "None", "Uniform", "Raw gap", "Entropy",
                "Value", r"$\Delta_{\mathrm U}$", r"$\Delta_{\mathrm E}$"),
        r"\midrule",
    ]
    for dataset, graph in DATASETS:
        for seed in STAGES[0][2]:
            row = [
                index["six_row", dataset, seed, "first_maximum", "exact", p]
                for p in ("no_lookup", "random", "raw_gap", "exact_entropy", "value")
            ]
            lines.append(tex_row(graph, seed, f"{int(row[0]['n']):,}",
                                 *[pct(item["rate"]) for item in row],
                                 *["$" + delta(item["delta_value"]) + "\\ " + interval(item) + "$"
                                   for item in (row[1], row[3])]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", "", r"\begin{table*}[!t]",
              r"\centering",
              r"\caption{Six-row selection gains, headroom, and fitted posterior concentration on aligned uniform blocks. The two ratios compare the gain of whole-block value over no lookup with the corresponding gain from a uniform lookup, under the fitted posterior or against the source mapping. $H$ is the hindsight ceiling under the common decoder. The headroom column reports the fraction $(A_{\rm value}-A_{\rm unif})/(H-A_{\rm unif})$ as a percentage. The final column is the mean posterior mass of a MAP permutation and all permutations one transposition away from it.}",
              r"\label{tab:six-posterior}",
              r"\ifdefined\JournalReviewCopy\footnotesize\else\small\fi",
              r"\setlength{\tabcolsep}{5pt}",
              r"\begin{tabular}{@{}lrrrrrr@{}}",
              r"\toprule",
              tex_row("Graph", "Split", "Fitted ratio", "Realised ratio", "$H$ (\\%)",
                      "Headroom (\\%)", "MAP or swap (\\%)"),
              r"\midrule"]
    for dataset, graph in DATASETS:
        for seed in STAGES[0][2]:
            item = diagnostic[dataset, seed]
            lines.append(tex_row(graph, seed, f"{float(item['model_gain_ratio']):.3f}",
                                 f"{float(item['realised_gain_ratio']):.3f}",
                                 pct(item["hindsight"]), pct(item["hindsight_fraction"]),
                                 pct(item["mean_within_one_swap"])))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return lines


def cell(row):
    if row is None:
        return r"\textemdash"
    rate = pct(row["rate"])
    if row["policy"] == "value":
        return r"\textbf{" + rate + "}"
    contrast = delta(row["delta_value"])
    bounds = interval(row)
    return rf"\shortstack[r]{{{rate} (\({contrast}\))\\ \({bounds}\)}}"


def policy_table(index, endpoint):
    suffix = "policy" if endpoint == "exact" else "row"
    outcome = "exact block recovery" if endpoint == "exact" else "fraction of correct rows"
    caption = (
        f"All-policy comparison for {outcome} on aligned uniform blocks. "
        r"Cells report the outcome rate (\%) and, in parentheses, the paired difference "
        r"of whole-block value minus the listed policy (percentage points), with a paired 95\% interval underneath. "
        r"Every $[0.00,0.00]$ interval shown corresponds to identical outcomes for the two policies on every block in that cell. "
        "The six-row comparison averages action ties and uses the same first-MAP decoder. "
        "The twelve-row comparison averages action and decoder ties. "
        "Sinkhorn entropy was evaluated in the twelve-row confirmation."
    )
    if endpoint == "row":
        caption += " The six-row hindsight diagnostic records exact recovery only."
    column_spec = r"{@{}ll*{4}{c}@{}}"
    header = [
        r"\toprule",
        tex_row("Policy", "Split", *[label for _, label in DATASETS]),
        r"\midrule",
    ]
    lines = []
    for stage, size, seeds, convention in STAGES:
        lines += [r"\begin{table}[!ht]", r"\centering"]
        if size == 6:
            lines.append(r"\caption{" + caption + r"}\suppmarker{tab:" + suffix + "-comparisons}")
        else:
            lines += [r"\ContinuedFloat", r"\caption{Twelve-row policy comparisons continued.}"]
        lines += [r"\small", r"\setlength{\tabcolsep}{5pt}",
                  r"\begin{tabular}" + column_spec, *header]
        lines.append(
            r"\multicolumn{6}{@{}l@{}}{\textbf{" + f"{size}-row blocks" + r"}}\\"
        )
        n = [index[stage, d, seeds[0], convention, endpoint, "value"]["n"] for d, _ in DATASETS]
        lines.append(tex_row("Blocks per split", "", *[f"{int(count):,}" for count in n]))
        lines.append(r"\addlinespace[3pt]")
        for policy, name in POLICIES:
            rows = [
                [index.get((stage, d, seed, convention, endpoint, policy)) for d, _ in DATASETS]
                for seed in seeds
            ]
            if all(row is None for seed_rows in rows for row in seed_rows):
                continue
            for i, (seed, seed_rows) in enumerate(zip(seeds, rows, strict=True)):
                lines.append(tex_row(name if i == 0 else "", seed, *map(cell, seed_rows)))
            lines.append(r"\addlinespace[2pt]")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\clearpage"]
    return lines


def main():
    results = indexed(read_csv("six_policy_comparisons.csv") + read_csv("twelve_policy_comparisons.csv"))
    diagnostic = {
        (r["dataset"], r["seed"]): r for r in read_csv("six_posterior_diagnostics.csv")
        if r["regime"] == "uniform" and r["condition"] == "aligned"
    }
    write_tex("six_summary_tables.tex", primary_results(results, diagnostic))
    write_tex("policy_comparisons.tex", policy_table(results, "exact") + [""] + policy_table(results, "row"))


if __name__ == "__main__":
    main()
