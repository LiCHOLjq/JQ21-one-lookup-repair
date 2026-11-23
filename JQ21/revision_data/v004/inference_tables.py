"""Render the supplementary control rates and the familywise comparison counts."""
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = {"cora":"Cora", "ogbn_arxiv_20k":r"\texttt{ogbn-arxiv}",
            "openalex_20k":"OpenAlex","amazon_computers":"Amazon Computers"}
LABELS = {"no_lookup":"No lookup", "random":"Uniform lookup", "raw_gap":"Raw-score gap",
          "exact_entropy":"Exact entropy", "expected_rows":"Expected rows",
          "least_confident":"Least confidence", "forced_gap":"Forced gap",
          "sinkhorn_entropy":"Sinkhorn entropy", "hindsight":"Hindsight ceiling"}
SIX = "first_maximum"
MATCHED = "action_tie_average"
TWELVE = "action_and_decoder_tie_average"


def load(name):
    with (HERE/name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def controls():
    records = {(r["dataset"],r["seed"],r["regime"],r["convention"],r["policy"]):r
               for r in load("inference_shuffled.csv")}
    lines = [r"\begin{table}[H]\centering\footnotesize",
             r"\caption{Exact block recovery (\%) after shuffling contexts. Uniform is the mean over all six possible lookup rows. Raw gap, entropy and value use the primary first-maximum convention. Expected rows, least confidence and forced gap use the matched convention of averaging tied actions. Under a uniform permutation the success probability after one truthful lookup is $1/120$ (0.83\%).}",
             r"\suppmarker{tab:shuffle}", r"\setlength{\tabcolsep}{2.8pt}",
             r"\begin{tabular}{@{}lrlrrrrrrrr@{}}\toprule",
             r"Graph & Split & Blocks & $n$ & Uniform & Raw gap & Entropy & Value & Exp.\ rows & Least conf. & Forced gap \\\midrule"]
    for dataset,label in DATASETS.items():
        for split in ("191126","201126"):
            for regime in ("uniform","banded"):
                rates=[]
                for conv,pol in [(SIX,"random"),(SIX,"raw_gap"),(SIX,"exact_entropy"),(SIX,"value"),
                                 (MATCHED,"expected_rows"),(MATCHED,"least_confident"),(MATCHED,"forced_gap")]:
                    row=records[dataset,split,regime,conv,pol]
                    rates.append(f"{100*float(row['rate']):.2f}")
                lines.append(" & ".join([label,split,regime.capitalize(),row["n"],*rates])+r" \\")
    return "\n".join(lines+[r"\bottomrule\end{tabular}",r"\end{table}",""])


def families():
    groups=defaultdict(list)
    for r in load("inference_tests.csv"):
        groups[r["convention"],r["endpoint"],r["policy"]].append(r)
    lines=[r"\section*{Multiplicity across the tested graph--split cells}",
           r"The post hoc tests use the same aligned blocks as the main comparisons. For each endpoint, convention, block size and comparator, we test the eight within-cell contrasts against whole-block value and apply Holm's step-down correction to those eight two-sided $p$-values. Conditional on a fixed graph, partition and fitted model, the sign-flip null treats the signs of nonzero paired block differences as independently exchangeable. Binary outcomes use the exact McNemar test. When all nonzero differences have the same magnitude, the exact binomial calculation also applies to fractional outcomes. For up to 18 unequal-magnitude differences we enumerate all signs. Otherwise we draw 199,999 random sign assignments and use the plus-one correction. Holm's procedure allows dependence among the eight cell tests. This analysis leaves dependence between blocks on a shared graph as a limitation. The reported error rate applies separately to each family.",
           r"\begin{table}[H]\centering\small",r"\caption{Counts of positive/negative contrasts for whole-block value among eight cells after Holm adjustment at 5\% familywise error. A dash indicates that the policy or endpoint was outside that comparison. The archive gives every cell estimate and both $p$-values.}",
           r"\suppmarker{tab:holm-families}",r"\begin{tabular}{@{}lrrrrrr@{}}\toprule",
           r"&\multicolumn{2}{c}{Six rows, primary}&\multicolumn{2}{c}{Six rows, matched}&\multicolumn{2}{c}{Twelve rows, matched}\\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
           r"Comparator & Exact & Row & Exact & Row & Exact & Row \\\midrule"]
    for policy,label in LABELS.items():
        cells=[]
        for conv in (SIX,MATCHED,TWELVE):
            for endpoint in ("exact","row"):
                rows=groups.get((conv,endpoint,policy))
                if rows:
                    assert len(rows)==8
                    diff=[float(r["delta"]) for r in rows if float(r["p_holm"])<=.05]
                    cells.append(f"{sum(v>0 for v in diff)}/{sum(v<0 for v in diff)}")
                else:
                    cells.append("---")
        lines.append(" & ".join([label,*cells])+r" \\")
    return "\n".join(lines+[r"\bottomrule\end{tabular}",r"\end{table}",""])


if __name__=="__main__":
    (HERE/"inference_S4.tex").write_text(controls(),encoding="utf-8")
    (HERE.parents[1]/"inference_supp.tex").write_text(families(),encoding="utf-8")

