"""Test paired six-row exact-recovery outcomes under the original first-maximum rule."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import binomtest


def main():
    here = Path(__file__).resolve().parent
    experiments = here.parents[1] / "research_direction_20261119" / "experiments"
    cells = defaultdict(Counter)
    for study in ("results", "transfer_results"):
        with (experiments / study / "blocks.csv").open(newline="", encoding="utf-8") as source:
            for block in csv.DictReader(source):
                if block["regime"] == "uniform" and block["control"] == "aligned":
                    pair = (int(block["value"]), int(block["entropy"]))
                    cells[block["dataset"], block["seed"]][pair] += 1

    assert len(cells) == 8
    results = []
    for (dataset, split), counts in sorted(cells.items()):
        value_only = counts[1, 0]
        entropy_only = counts[0, 1]
        discordant = value_only + entropy_only
        p = binomtest(value_only, discordant, 0.5).pvalue if discordant else 1.0
        results.append(dict(dataset=dataset, split=split, blocks=sum(counts.values()),
                            value_only=value_only, entropy_only=entropy_only, p_exact=p))

    adjusted = 0.0
    for rank, row in enumerate(sorted(results, key=lambda row: row["p_exact"])):
        adjusted = max(adjusted, (len(results) - rank) * row["p_exact"])
        row["p_holm"] = min(adjusted, 1.0)
    with (here / "six_entropy_holm.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
