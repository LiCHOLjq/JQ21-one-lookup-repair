"""Replay matched twelve-row comparisons on the frozen confirmation blocks.

Build the sibling kernel with a C++17 compiler, then pass its shared library
using --kernel. Examples:
  cl /nologo /O2 /EHsc /std:c++17 /LD twelve_policy_kernel.cpp
  c++ -O3 -std=c++17 -shared -fPIC twelve_policy_kernel.cpp -o twelve_policy_kernel.so
"""

import argparse
import csv
import ctypes
import json
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp


STAGE = "twelve_row"
CONVENTION = "action_and_decoder_tie_average"
POLICIES = [
    "no_lookup", "random", "raw_gap", "exact_entropy", "value",
    "expected_rows", "least_confident", "forced_gap", "sinkhorn_entropy", "hindsight",
]


class Kernel:
    def __init__(self, path):
        library = ctypes.CDLL(str(path.resolve()))
        doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        integers = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        library.expected_rows.argtypes = [doubles, ctypes.c_int, doubles, doubles]
        library.expected_rows.restype = None
        library.decoder_accuracy.argtypes = [doubles, integers, ctypes.c_int,
                                            ctypes.c_double, doubles, integers, doubles]
        library.decoder_accuracy.restype = None
        self.lookahead = library.expected_rows
        self.decoder = library.decoder_accuracy

    def row_values(self, score, temperature):
        n = len(score)
        _, assignment = linear_sum_assignment(score, maximize=True)
        loss = score[np.arange(n), assignment][:, None] - score[:, assignment]
        # A shortest-path potential in the exchange graph reweights edges
        # without changing the relative weight of any complete permutation.
        potential = np.zeros(n)
        for _ in range(n - 1):
            potential = np.minimum(potential, (potential[:, None] + loss).min(0))
        reduced = loss + potential[:, None] - potential[None, :]
        assert reduced.min() > -1e-12
        weights = np.empty_like(score)
        weights[:, assignment] = np.exp(-np.maximum(reduced, 0.0) / temperature)
        value = np.empty(n)
        marginals = np.zeros_like(score)
        self.lookahead(weights, n, value, marginals)
        return value, marginals

    def row_accuracy(self, score, truth):
        n = len(score)
        accuracy = np.empty(n + 1)
        counts = np.empty(n + 1, dtype=np.int64)
        optima = np.empty(n + 1)
        self.decoder(score, truth, n, 1e-12, accuracy, counts, optima)
        return accuracy, counts, optima


def validate(kernel):
    rng = np.random.default_rng(391119)
    for n in range(2, 8):
        options = np.array(list(permutations(range(n))))
        rows = np.arange(n)
        for score in [rng.normal(size=(n, n)), rng.integers(-2, 3, (n, n)).astype(float),
                      np.zeros((n, n))]:
            truth = rng.permutation(n)
            assignment_score = score[rows, options].sum(1)
            for temperature in (0.02, 0.9):
                log_weights = assignment_score / temperature
                probability = np.exp(log_weights - logsumexp(log_weights))
                brute = np.ones(n)
                for i in rows:
                    for k in rows[rows != i]:
                        bins = n * options[:, i] + options[:, k]
                        joint = np.bincount(bins, weights=probability, minlength=n*n)
                        brute[i] += joint.reshape(n, n).max(1).sum()
                value, _ = kernel.row_values(score, temperature)
                np.testing.assert_allclose(value, brute, rtol=0, atol=1e-11)
            accuracy, counts, optima = kernel.row_accuracy(score, truth)
            for i in range(n + 1):
                keep = options[:, i] == truth[i] if i < n else np.ones(len(options), bool)
                optimum = np.max(assignment_score[keep])
                tied = keep & np.isclose(assignment_score, optimum, atol=1e-12, rtol=0)
                assert counts[i] == tied.sum()
                np.testing.assert_allclose(optima[i], optimum, rtol=0, atol=1e-12)
                np.testing.assert_allclose(accuracy[i], (options[tied] == truth).mean(),
                                           rtol=0, atol=1e-12)


def tied_actions(score):
    tied = np.isclose(score, score.max(), rtol=1e-10, atol=1e-12)
    return tied / tied.sum()


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def paired_intervals(outcomes, seed):
    rng = np.random.default_rng(seed + 440121)
    delta = outcomes[:, [POLICIES.index("value")]] - outcomes
    n = len(delta)
    draws = []
    for _ in range(10):
        indices = rng.integers(0, n, (1_000, n))
        draws.append(delta[indices].mean(1))
    return delta.mean(0), np.quantile(np.concatenate(draws), [0.025, 0.975], axis=0)


def evaluate(score, truth, marginal, forced_counts, action_success, choices, block, kernel,
             temperature):
    rows = np.arange(len(score))
    row_value, replay_marginal = kernel.row_values(score, temperature)
    marginal_error = np.max(np.abs(marginal - replay_marginal))
    assert marginal_error < 1e-10, (block["dataset"], block["seed"], block["block"], marginal_error)
    accuracy, counts, optima = kernel.row_accuracy(score, truth)
    np.testing.assert_array_equal(counts[:-1], forced_counts[rows, truth])
    assert counts[-1] == int(block["no_lookup_decoder_ties"])
    true_score = score[rows, truth].sum()
    exact = np.isclose(true_score, optima, rtol=0, atol=1e-12) / counts
    np.testing.assert_array_equal(exact[:-1], action_success)
    assert exact[-1] == float(block["no_lookup"])
    # Include the one certainly correct queried row, as in the six-row replay.
    scores_sorted = np.sort(score, axis=1)
    policies = {
        "exact_entropy": choices[0],
        "sinkhorn_entropy": choices[1],
        "value": choices[2],
        "random": choices[3],
        "forced_gap": choices[4],
        "raw_gap": tied_actions(scores_sorted[:, -2] - scores_sorted[:, -1]),
        "expected_rows": tied_actions(row_value),
        "least_confident": tied_actions(-marginal.max(1)),
    }
    exact_rates = {p: w @ exact[:-1] for p, w in policies.items()}
    row_rates = {p: w @ accuracy[:-1] for p, w in policies.items()}
    exact_rates.update(no_lookup=exact[-1], hindsight=exact[:-1].max())
    row_rates.update(no_lookup=accuracy[-1], hindsight=accuracy[:-1].max())
    for p, source in [(p, p) for p in policies if p in block] + [("hindsight", "oracle")]:
        np.testing.assert_allclose(exact_rates[p], float(block[source]), rtol=0, atol=1e-14)
    return exact_rates, row_rates, marginal_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, required=True)
    here = Path(__file__).resolve().parent
    source = here.parent.parent / "research_direction_20261119/experiments/b12_confirmation"
    parser.add_argument("--frozen", type=Path, default=source / "accelerated_results")
    parser.add_argument("--out", type=Path, default=here)
    args = parser.parse_args()
    kernel = Kernel(args.kernel)
    validate(kernel)
    metadata = json.loads((args.frozen / "run.json").read_text(encoding="utf-8"))
    blocks = defaultdict(list)
    for row in read_csv(args.frozen / "blocks.csv"):
        blocks[row["dataset"], int(row["seed"])].append(row)

    run_results = []
    policy_results = []
    headroom = []
    overlap_sets = defaultdict(list)
    max_marginal_error = 0.0
    for run in metadata["runs"]:
        dataset, seed = run["dataset"], run["seed"]
        cell = blocks[dataset, seed]
        path = args.frozen / f"{dataset}_{seed}_raw.npz"
        with np.load(path, allow_pickle=False) as archive:
            score, truth, marginal, counts, exact, choices = (
                archive[key] for key in ["score", "truth", "true_marginals", "forced_counts",
                                         "action_success", "policy_action_probabilities"]
            )
            ids = set(n for row in cell for n in map(int, row["vertices"].split()))
            test = set(archive["test_vertices"])
            overlap_sets[dataset].append((seed, test, ids))
        assert run["blocks"] == len(score) == len(cell)
        for i, block in enumerate(cell):
            assert i == int(block["block"])
            vertices = np.fromstring(block["vertices"], sep=" ", dtype=np.int64)
            slots = np.fromstring(block["feature_slots"], sep=" ", dtype=np.int64)
            np.testing.assert_array_equal(slots[truth[i]], vertices)
            exact_rates, row_rates, error = evaluate(
                score[i], truth[i], marginal[i], counts[i], exact[i], choices[i],
                block, kernel, run["temperature"],
            )
            max_marginal_error = max(max_marginal_error, error)
            run_results.append({
                "dataset": dataset, "seed": seed, "block": i,
                **{f"exact_{p}": r for p, r in exact_rates.items()},
                **{f"row_{p}": r for p, r in row_rates.items()},
            })
        records = run_results[-len(cell):]
        for endpoint in ("exact", "row"):
            outcomes = np.array([[row[f"{endpoint}_{p}"] for p in POLICIES] for row in records])
            deltas, bounds = paired_intervals(outcomes, seed)
            for i, name in enumerate(POLICIES):
                policy_results.append({
                    "stage": STAGE, "dataset": dataset, "seed": seed, "block_size": 12,
                    "regime": "uniform", "condition": "aligned", "convention": CONVENTION,
                    "endpoint": endpoint, "policy": name, "n": len(cell),
                    "outcome_sum": outcomes[:, i].sum(), "rate": outcomes[:, i].mean(),
                    "delta_value": deltas[i], "ci_low": bounds[0, i], "ci_high": bounds[1, i],
                })
        mean = {name: np.mean([row[f"exact_{name}"] for row in records]) for name in POLICIES}
        available = mean["hindsight"] - mean["random"]
        gain = mean["value"] - mean["random"]
        headroom.append({
            "dataset": dataset, "seed": seed, "n": len(cell),
            "no_lookup": mean["no_lookup"], "random": mean["random"], "value": mean["value"],
            "hindsight": mean["hindsight"], "difference": gain, "headroom": available,
            "share_of_headroom": gain / available if available > 0 else "",
        })
        print(f"{dataset} {seed}: {100 * gain:+.2f} pp; hindsight {100 * mean['hindsight']:.2f}%",
              flush=True)

    overlap = []
    for dataset, cells in overlap_sets.items():
        (sa, ta, ea), (sb, tb, eb) = sorted(cells)
        assert len(ta) == len(tb) and len(ea) == len(eb)
        overlap.append({
            "dataset": dataset, "seed_a": sa, "seed_b": sb,
            "evaluation_fold_per_split": len(ta), "evaluation_fold_shared": len(ta & tb),
            "evaluation_fold_overlap": len(ta & tb) / len(ta),
            "evaluated_per_split": len(ea), "evaluated_shared": len(ea & eb),
            "evaluated_overlap": len(ea & eb) / len(ea),
        })

    frozen_comparison = read_csv(args.frozen / "paired_summary.csv")
    new_results = {(r["dataset"], r["seed"], r["policy"]): r for r in policy_results
                   if r["endpoint"] == "exact"}
    for row in frozen_comparison:
        if row["policy"] == "value":
            replay = new_results[row["dataset"], int(row["seed"]), row["comparator"]]
            np.testing.assert_allclose(
                [float(row[k]) for k in ["difference", "ci_low", "ci_high"]],
                [replay[k] for k in ["delta_value", "ci_low", "ci_high"]], rtol=0, atol=1e-14,
            )
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "twelve_policy_blocks.csv", run_results)
    write_csv(args.out / "twelve_policy_comparisons.csv", policy_results)
    write_csv(args.out / "twelve_headroom.csv", headroom)
    write_csv(args.out / "twelve_vertex_overlap.csv", overlap)
    print(f"Checked all {len(run_results)} blocks. Max replay marginal error {max_marginal_error:.3g}.")


if __name__ == "__main__":
    main()
