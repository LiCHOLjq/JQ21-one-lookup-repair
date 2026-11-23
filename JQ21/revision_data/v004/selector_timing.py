"""Time two Python/SciPy selectors on matched archived twelve-row scores."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from time import perf_counter_ns

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from scipy.special import xlogy


def main():
    here = Path(__file__).resolve().parent
    exp = here.parent.parent.parent / "research_direction_20261119" / "experiments"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=exp)
    parser.add_argument("--revision-data", type=Path, default=here.parent)
    args = parser.parse_args()
    sys.path[:0] = [str(args.experiments), str(args.revision_data)]
    from block_stress import permanent
    from sensitivity_benchmark import exchange_loss

    def value(matrix, temperature):
        losses = exchange_loss(matrix)
        # Remove one MAP contribution from each row before summing the gains.
        losses[np.arange(len(matrix)), losses.argmin(axis=1)] = np.inf
        return np.exp(-losses / temperature).sum(axis=1).argmax()

    def entropy(matrix, temperature):
        _, _, probs = permanent(matrix, np.array([temperature]), marginals=True)
        return (-xlogy(probs, probs)).sum(axis=1).argmax()

    base = args.experiments / "b12_confirmation" / "accelerated_results"
    rows = []
    for cell in json.loads((base / "run.json").read_text())["runs"]:
        graph, split, temp = cell["dataset"], cell["seed"], cell["temperature"]
        with np.load(base / f"{graph}_{split}_raw.npz") as raw:
            matrices = raw["score"][:16]
            _, _, check = permanent(matrices[0], np.array([temp]), marginals=True)
            np.testing.assert_allclose(check, raw["true_marginals"][0], atol=1e-10)
            np.testing.assert_allclose(exchange_loss(matrices[0]), raw["forced_loss"][0], atol=1e-10)
        for method in (value, entropy):
            method(matrices[0], temp)
        for repeat in range(7):
            for method in ((value, entropy) if repeat % 2 == 0 else (entropy, value)):
                start = perf_counter_ns()
                for matrix in matrices:
                    method(matrix, temp)
                ns = perf_counter_ns() - start
                rows.append(dict(graph=graph, split=split, blocks=len(matrices),
                                 repeat=repeat, selector=method.__name__, ns_per_block=ns/len(matrices)))
    with (here / "selector_timing.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
