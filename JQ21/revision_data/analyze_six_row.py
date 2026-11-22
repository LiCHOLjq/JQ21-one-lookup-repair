"""Replay saved six-row blocks and produce paired policy and posterior diagnostics."""

import argparse
import csv
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.special import logsumexp


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PRIMARY_SEEDS = (191126, 201126)
CONFIRMATION_SEEDS = (119003, 119004)
PRIMARY_KEYS = {
    "no_lookup": "unqueried",
    "random": "random",
    "raw_gap": "gap",
    "exact_entropy": "entropy",
    "value": "value",
    "hindsight": "oracle",
}
SUPPLEMENTAL_KEYS = {
    "exact_entropy": "marginal_entropy",
    "value": "whole_value",
    "expected_rows": "expected_rows",
    "least_confident": "least_confident",
    "forced_gap": "forced_runner_up",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def group(rows, columns):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[column] for column in columns)].append(row)
    return groups


def score_block(score, truth, temperature, permutations, pair_ids):
    b = len(score)
    row_ids = np.arange(b)
    joint = score[row_ids, permutations].sum(axis=1)
    best_index = np.argmax(joint)
    sigma = permutations[best_index]
    log_weight = (joint - joint[best_index]) / temperature
    probabilities = np.exp(log_weight - logsumexp(log_weight))
    within_swap = np.count_nonzero(permutations != sigma, axis=1) <= 2
    marginal = np.stack([
        np.bincount(permutations[:, i], weights=probabilities, minlength=b)
        for i in row_ids
    ])
    forced_log_weight = np.empty((b, b))
    residual_correct = np.zeros(b)
    matches = np.empty((b, b), dtype=bool)
    for i in row_ids:
        for j in row_ids:
            indices = np.flatnonzero(permutations[:, i] == j)
            completion = indices[np.argmax(joint[indices])]
            forced_log_weight[i, j] = log_weight[completion]
            if j == truth[i]:
                matches[i] = permutations[completion] == truth
        for k in row_ids:
            joint_table = np.bincount(
                pair_ids[i, k], weights=probabilities, minlength=b * b
            ).reshape(b, b)
            residual_correct[i] += joint_table.max(axis=1).sum()

    forced_mass = probabilities[best_index] * np.exp(forced_log_weight)
    sorted_raw = np.sort(score, axis=1)
    objectives = {
        "raw_gap": sorted_raw[:, -2] - sorted_raw[:, -1],
        "exact_entropy": -(marginal * np.log(np.maximum(marginal, 1e-300))).sum(axis=1),
        "value": forced_mass.sum(axis=1),
        "expected_rows": residual_correct,
        "least_confident": -marginal.max(axis=1),
        "forced_gap": np.sort(forced_mass, axis=1)[:, -2],
    }
    result = {}
    for policy, objective in objectives.items():
        tied = np.isclose(objective, objective.max(), rtol=1e-11, atol=1e-15)
        result[policy] = (matches[tied].all(axis=1).mean(), matches[tied].mean())

    forced_log_weight[row_ids, sigma] = -np.inf
    gains = probabilities[best_index] * np.exp(logsumexp(forced_log_weight, axis=1))
    diagnostics = {
        "best_gain": float(gains.max()),
        "random_gain": float(gains.mean()),
        "within_one_swap": float(probabilities[within_swap].sum()),
        "swap_only": float(probabilities[within_swap].sum() - probabilities[best_index]),
        "tied_raw_gap": int(np.isclose(
            objectives["raw_gap"], objectives["raw_gap"].max(), rtol=1e-11, atol=1e-15
        ).sum() > 1),
    }
    assert gains.mean() + 1e-12 >= (2 / b) * gains.max()
    return result, diagnostics


def metrics(row, replayed, supplemental=None):
    primary = {
        policy: (float(row[key]), float(row["rows_" + key]) if policy != "hindsight" else None)
        for policy, key in PRIMARY_KEYS.items()
    }
    tied = primary | replayed
    if supplemental is not None:
        for policy, key in SUPPLEMENTAL_KEYS.items():
            tied[policy] = (float(supplemental[key]), float(supplemental["rows_" + key]))
    return {"first_maximum": primary, "action_tie_average": tied}


def replay(experiments, historical):
    sys.path.insert(0, str(experiments))
    study = importlib.import_module("study")
    audit = importlib.import_module("numerical_audit")
    permutations = study.PERMS
    pair_ids = permutations.T[:, None, :] * study.B + permutations.T[None, :, :]
    output = {}
    graph_sizes = {}
    for folder in ("results", "transfer_results"):
        base = experiments / folder
        raw_groups = group(read_csv(base / "blocks.csv"), ("dataset", "seed"))
        supplemental = {
            (r["dataset"], r["seed"], r["regime"], r["block"]): r
            for r in read_csv(base / "secondary.csv")
        }
        for fit in json.loads((base / "run.json").read_text(encoding="utf-8"))["runs"]:
            dataset, seed = fit["dataset"], fit["seed"]
            graph_sizes[dataset] = fit["n"]
            raw = raw_groups[dataset, str(seed)]
            x, edges, source, kind = audit.read_inputs(experiments, dataset, historical)
            if kind == "frozen_public_cache":
                assert audit.sha256(source) == fit["sha256"], source
            assert x.shape == (fit["n"], fit["d"])
            rng = np.random.default_rng(seed)
            anchors, calibration, test = np.split(
                rng.permutation(len(x)), [int(.55 * len(x)), int(.70 * len(x))]
            )
            assert (len(anchors), len(calibration), len(test)) == (
                fit["anchor_n"], fit["cal_n"], fit["test_n"]
            )
            x, contexts, _ = study.predictions(x, edges, anchors)
            h = contexts["centroid"]
            prediction = contexts[fit["calibration_method"]]
            direction = rng.normal(size=x.shape[1])
            for _ in study.blocks(calibration, h, "uniform", rng, direction):
                pass  # Advance through the random draws in the original calibration.
            index = 0
            for regime in ("uniform", "banded"):
                for block, (vertices, slots, truth) in enumerate(
                    study.blocks(test, h, regime, rng, direction)
                ):
                    s = prediction[vertices] @ x[slots].T
                    for condition, score in (("aligned", s), ("shuffled", s[rng.permutation(study.B)])):
                        row = raw[index]
                        index += 1
                        assert (row["regime"], row["control"], int(row["block"])) == (
                            regime, condition, block
                        )
                        assert np.array_equal(np.fromstring(row["vertices"], sep=" ", dtype=int), vertices)
                        assert np.array_equal(np.fromstring(row["feature_slots"], sep=" ", dtype=int), slots)
                        saved = supplemental.get((dataset, str(seed), regime, str(block))) if condition == "aligned" else None
                        policies, diagnostics = score_block(
                            score, truth, fit["temperature"], permutations, pair_ids
                        )
                        if saved is not None:
                            assert abs(diagnostics["best_gain"] - float(saved["best_model_gain"])) < 5e-12
                            assert abs(diagnostics["random_gain"] - float(saved["random_model_gain"])) < 5e-12
                        cell = (dataset, seed, regime, condition)
                        output.setdefault(cell, []).append((metrics(row, policies, saved), diagnostics))
            assert index == len(raw)
            print(dataset, seed, len(raw), "blocks replayed", flush=True)
    return output, graph_sizes


def bootstrap_weights(rng, n, draws=4_000):
    return np.stack([
        np.bincount(draw, minlength=n) for draw in rng.integers(0, n, size=(draws, n))
    ]) / n


def summaries(cells):
    frozen_rng = np.random.default_rng(76119)
    additional_rng = np.random.default_rng(76120)
    policy_rows = []
    diagnostic_rows = []
    for (dataset, seed, regime, condition), cell in cells.items():
        n = len(cell)
        estimates, diagnostics = zip(*cell)
        weights = bootstrap_weights(additional_rng, n)
        published_weights = {
            policy: bootstrap_weights(frozen_rng, n) for policy in ("random", "exact_entropy")
        }
        entry = dict(stage="six_row", dataset=dataset, seed=seed, block_size=6,
                     regime=regime, condition=condition)
        for convention, template in estimates[0].items():
            for outcome_index, endpoint in enumerate(("exact", "row")):
                policies = [p for p, values in template.items() if values[outcome_index] is not None]
                scores = np.array([
                    [item[convention][p][outcome_index] for p in policies] for item in estimates
                ])
                value = scores[:, policies.index("value")]
                differences = value[:, None] - scores
                means = scores.mean(axis=0)
                contrasts = differences.mean(axis=0)
                lows, highs = np.quantile(weights @ differences, [0.025, 0.975], axis=0)
                for j, policy in enumerate(policies):
                    if convention == "first_maximum" and endpoint == "exact" and policy in published_weights:
                        lows[j], highs[j] = np.quantile(
                            published_weights[policy] @ differences[:, j], [0.025, 0.975]
                        )
                    policy_rows.append(entry | dict(
                        convention=convention, endpoint=endpoint, policy=policy, n=n,
                        outcome_sum=scores[:, j].sum(), rate=means[j], delta_value=contrasts[j],
                        ci_low=lows[j], ci_high=highs[j],
                    ))
        means = {
            policy: np.mean([item["first_maximum"][policy][0] for item in estimates])
            for policy in ("no_lookup", "random", "value", "hindsight")
        }
        best_gain = np.mean([item["best_gain"] for item in diagnostics])
        random_gain = np.mean([item["random_gain"] for item in diagnostics])
        realised_denominator = means["random"] - means["no_lookup"]
        headroom = means["hindsight"] - means["random"]
        diagnostic_rows.append(entry | dict(
            n=n, **means, model_best_gain=best_gain, model_random_gain=random_gain,
            model_gain_ratio=best_gain / random_gain,
            realised_gain_ratio=(means["value"] - means["no_lookup"]) / realised_denominator
                if realised_denominator > 0 else "",
            hindsight_headroom=headroom,
            hindsight_fraction=(means["value"] - means["random"]) / headroom if headroom > 0 else "",
            perfect_headroom_fraction=(means["value"] - means["random"]) / (1 - means["random"]),
            mean_within_one_swap=np.mean([item["within_one_swap"] for item in diagnostics]),
            mean_swap_only=np.mean([item["swap_only"] for item in diagnostics]),
            numerically_tied_raw_gap_count=sum(item["tied_raw_gap"] for item in diagnostics),
        ))
    return policy_rows, diagnostic_rows


def overlap(experiments, sizes):
    all_six = [row for folder in ("results", "transfer_results")
               for row in read_csv(experiments / folder / "blocks.csv")
               if row["regime"] == "uniform" and row["control"] == "aligned"]
    all_twelve = read_csv(experiments / "b12_confirmation/accelerated_results/blocks.csv")
    rows = []
    for stage, seeds, blocks in (
        ("six_row", PRIMARY_SEEDS, all_six),
        ("twelve_row", CONFIRMATION_SEEDS, all_twelve),
    ):
        block_groups = group(blocks, ("dataset", "seed"))
        for dataset, n in sizes.items():
            for population in ("evaluation_fold", "evaluated_blocks"):
                vertices = []
                for seed in seeds:
                    if population == "evaluation_fold":
                        ids = np.random.default_rng(seed).permutation(n)[int(.70 * n):]
                    else:
                        ids = [int(v) for block in block_groups[dataset, str(seed)]
                               for v in block["vertices"].split()]
                        assert len(ids) == len(set(ids))
                    vertices.append(set(ids))
                count = len(vertices[0] & vertices[1])
                rows.append(dict(stage=stage, dataset=dataset, population=population,
                                 seed_a=seeds[0], seed_b=seeds[1], n_a=len(vertices[0]),
                                 n_b=len(vertices[1]), overlap=count,
                                 fraction_of_seed_a=count / len(vertices[0]),
                                 fraction_of_seed_b=count / len(vertices[1])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path,
                        default=WORKSPACE / "research_direction_20261119" / "experiments")
    parser.add_argument("--historical", type=Path, default=WORKSPACE / "research_discovery_20261115")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cells, sizes = replay(args.experiments, args.historical)
    policy, diagnostics = summaries(cells)
    write_csv(args.out / "six_policy_comparisons.csv", policy)
    write_csv(args.out / "six_posterior_diagnostics.csv", diagnostics)
    write_csv(args.out / "evaluation_split_overlap.csv", overlap(args.experiments, sizes))


if __name__ == "__main__":
    main()
