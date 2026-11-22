"""Local sensitivity audit; does not replace or modify the frozen experiment.

Replay the eight aligned/uniform six-row cells from their frozen IDs, column
slots, scorer choice and temperature. Refit the deterministic anchor predictor
only; never recalibrate or regenerate requests. The three historical files are
used only as public feature/edge caches. Amazon is transformed in memory with
the exact recipe in transfer.py.

The stable whole-block objective explicitly excludes one MAP cell from each
row and computes logsumexp of the remaining forced-completion log weights.
Numerically tied actions are averaged uniformly. Log-gain ties use absolute
distance 1e-11 (roughly relative gain tolerance 1e-11), with no tolerance on the
large common MAP-success constant. Entropy/gap ties use the supplementary
policy tolerance rtol=1e-11, atol=1e-15. All variants share the original
lexicographic permutation-MAP decoder. Frozen outcomes remain in the CSV so
platform/replay changes are distinguishable from selection/tie changes.

Run locally: python -B numerical_audit.py --out numerical_audit.csv
Dependencies: NumPy and SciPy. No network access, server access or package
installation. Only the specified output CSV is written.
"""
import os

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from study import B, ROWS, PERMS, evaluate, exchange, predictions

LOG_GAIN_ATOL = 1e-11
POLICY_RTOL = 1e-11
POLICY_ATOL = 1e-15


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_inputs(base, dataset, historical):
    if dataset == "amazon_computers":
        path = base / "public_data/amazon_original.npz"
        with np.load(path, allow_pickle=False) as raw:
            def load(prefix):
                return csr_matrix(
                    (raw[prefix + "_data"], raw[prefix + "_indices"],
                     raw[prefix + "_indptr"]),
                    shape=tuple(raw[prefix + "_shape"]))
            x = load("attr").toarray().astype("float32").astype(float)
            edges = np.column_stack(load("adj").nonzero()).astype(int)
        expected = json.loads((base / "public_data/provenance.json").read_text())["sha256"]
        assert sha256(path).lower() == expected.lower(), "Amazon source digest differs"
        return x, edges, path, "original_sparse_npz_converted_in_memory"
    folder = "extra_data" if dataset == "openalex_20k" else "data"
    path = historical / folder / (dataset + ".npz")
    with np.load(path, allow_pickle=False) as raw:
        return raw["x"].astype(float), raw["edges"].astype(int), path, "frozen_public_cache"


def audit_block(score, truth, temperature):
    joint = score[ROWS, PERMS].sum(1)
    best = int(np.argmax(joint))
    sigma = PERMS[best]
    forced = np.array([[joint[PERMS[:, i] == j].max() for j in ROWS]
                       for i in ROWS])
    log_terms = (forced - joint[best]) / temperature
    log_terms[ROWS, sigma] = -np.inf  # Remove exactly one common MAP term.
    log_gain = logsumexp(log_terms, axis=1)
    stable_top = (log_gain.max() - log_gain) <= LOG_GAIN_ATOL

    # Independent exchange-path calculation must agree with exhaustive maxima.
    loss = exchange(score)
    exchange_terms = -loss / temperature
    exchange_terms[ROWS, sigma] = -np.inf
    exchange_log_gain = logsumexp(exchange_terms, axis=1)
    discrepancy = float(np.max(np.abs(exchange_log_gain - log_gain)))
    assert discrepancy < 1e-9, "Exchange/enumeration disagreement"

    mass = np.exp((joint - joint[best]) / temperature)
    mass /= mass.sum()
    marg = np.array([np.bincount(PERMS[:, i], weights=mass, minlength=B)
                     for i in ROWS])
    entropy = -(marg * np.log(np.maximum(marg, 1e-300))).sum(1)
    sorted_score = np.sort(score, axis=1)
    negative_gap = sorted_score[:, -2] - sorted_score[:, -1]
    entropy_top = np.isclose(entropy, entropy.max(), rtol=POLICY_RTOL, atol=POLICY_ATOL)
    gap_top = np.isclose(negative_gap, negative_gap.max(), rtol=POLICY_RTOL, atol=POLICY_ATOL)

    matches = []
    for i in ROWS:
        candidates = np.flatnonzero(PERMS[:, i] == truth[i])
        decoded = PERMS[candidates[np.argmax(joint[candidates])]]
        matches.append(decoded == truth)
    matches = np.array(matches)
    ok = matches.all(1)
    legacy = evaluate(score, truth, temperature)
    assert np.isclose(ok.mean(), legacy["random"], rtol=0, atol=1e-15)
    exact_first = int(np.argmax(log_gain))
    return {
        "legacy": legacy,
        "stable_first": float(ok[exact_first]),
        "stable_value": float(ok[stable_top].mean()),
        "stable_rows_value": float(matches[stable_top].mean()),
        "tied_entropy": float(ok[entropy_top].mean()),
        "tied_gap": float(ok[gap_top].mean()),
        "stable_first_action_changed": int(exact_first != legacy["asked_value"]),
        "legacy_action_outside_stable_ties": int(not stable_top[legacy["asked_value"]]),
        "stable_tied": int(stable_top.sum() > 1),
        "entropy_tied": int(entropy_top.sum() > 1),
        "exchange_log_gain_error": discrepancy,
    }


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=base / "numerical_audit.csv")
    parser.add_argument("--historical", type=Path,
                        default=base.parents[1] / "research_discovery_20261115")
    args = parser.parse_args()
    summaries = []
    for frozen_folder in ("results", "transfer_results"):
        run_path = base / frozen_folder / "run.json"
        blocks_path = base / frozen_folder / "blocks.csv"
        meta = json.loads(run_path.read_text())["runs"]
        with blocks_path.open(newline="") as stream:
            frozen = [r for r in csv.DictReader(stream)
                      if r["regime"] == "uniform" and r["control"] == "aligned"]
        for m in meta:
            dataset, seed = m["dataset"], m["seed"]
            records = [r for r in frozen if r["dataset"] == dataset and int(r["seed"]) == seed]
            x, edges, input_path, input_kind = read_inputs(base, dataset, args.historical)
            input_digest = sha256(input_path)
            if input_kind == "frozen_public_cache":
                assert input_digest == m["sha256"], "Frozen public-cache digest differs"
            assert x.shape == (m["n"], m["d"])
            order = np.random.default_rng(seed).permutation(len(x))
            anchors = order[:int(.55 * len(x))]
            assert len(anchors) == m["anchor_n"]
            test = set(order[int(.70 * len(x)):].tolist())
            x, contexts, _counts = predictions(x, edges, anchors)
            pred = contexts[m["calibration_method"]]
            audited = []
            for r in records:
                ids = np.fromstring(r["vertices"], sep=" ", dtype=int)
                cols = np.fromstring(r["feature_slots"], sep=" ", dtype=int)
                assert len(ids) == B and len(set(ids)) == B and set(ids) == set(cols)
                assert set(ids).issubset(test)
                truth = np.argmax(ids[:, None] == cols[None, :], axis=1)
                audited.append(audit_block(pred[ids] @ x[cols].T, truth, m["temperature"]))
            mean = lambda key: float(np.mean([a[key] for a in audited]))
            frozen_mean = lambda key: float(np.mean([float(r[key]) for r in records]))
            replay_mean = lambda key: float(np.mean([a["legacy"][key] for a in audited]))
            count = lambda key: int(sum(a[key] for a in audited))
            replay_mismatches = lambda key: sum(
                not np.isclose(a["legacy"][key], float(r[key]), rtol=0, atol=1e-14)
                for a, r in zip(audited, records))
            row = dict(dataset=dataset, seed=seed, blocks=len(records),
                       calibration_method=m["calibration_method"], temperature=m["temperature"],
                       frozen_value=frozen_mean("value"), local_legacy_value=replay_mean("value"),
                       stable_first_value=mean("stable_first"), stable_tie_value=mean("stable_value"),
                       frozen_entropy=frozen_mean("entropy"), local_legacy_entropy=replay_mean("entropy"),
                       tied_entropy=mean("tied_entropy"), frozen_gap=frozen_mean("gap"),
                       tied_gap=mean("tied_gap"), frozen_random=frozen_mean("random"),
                       local_random=replay_mean("random"), frozen_no_lookup=frozen_mean("unqueried"),
                       local_no_lookup=replay_mean("unqueried"),
                       stable_rows_value=mean("stable_rows_value"),
                       stable_minus_frozen_value_pp=100 * (mean("stable_value") - frozen_mean("value")),
                       stable_minus_local_legacy_value_pp=100 * (mean("stable_value") - replay_mean("value")),
                       stable_gain_over_random_pp=100 * (mean("stable_value") - replay_mean("random")),
                       frozen_gain_over_random_pp=100 * (frozen_mean("value") - frozen_mean("random")),
                       frozen_passes_5pp=int(frozen_mean("value") - frozen_mean("random") >= .05),
                       stable_passes_5pp=int(mean("stable_value") - replay_mean("random") >= .05),
                       stable_first_action_changed=count("stable_first_action_changed"),
                       legacy_action_outside_stable_ties=count("legacy_action_outside_stable_ties"),
                       stable_tied_blocks=count("stable_tied"), entropy_tied_blocks=count("entropy_tied"),
                       replay_value_mismatch_blocks=replay_mismatches("value"),
                       replay_entropy_mismatch_blocks=replay_mismatches("entropy"),
                       replay_random_mismatch_blocks=replay_mismatches("random"),
                       replay_no_lookup_mismatch_blocks=replay_mismatches("unqueried"),
                       max_exchange_log_gain_error=max(a["exchange_log_gain_error"] for a in audited),
                       log_gain_tie_atol=LOG_GAIN_ATOL, policy_tie_rtol=POLICY_RTOL,
                       policy_tie_atol=POLICY_ATOL, input_kind=input_kind, input_path=str(input_path),
                       input_sha256=input_digest, frozen_input_sha256=m["sha256"],
                       frozen_blocks_sha256=sha256(blocks_path), frozen_run_sha256=sha256(run_path),
                       study_sha256=sha256(base / "study.py"), audit_sha256=sha256(Path(__file__)),
                       python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
            summaries.append(row)
            print(dataset, seed, "n", len(records), "frozen", round(row["frozen_value"], 6),
                  "stable", round(row["stable_tie_value"], 6),
                  "delta_pp", round(row["stable_minus_frozen_value_pp"], 6),
                  "gain_pp", round(row["stable_gain_over_random_pp"], 6),
                  "replay_mismatch", row["replay_value_mismatch_blocks"], flush=True)
            del contexts, pred, x, edges, audited
    assert len(summaries) == 8
    with args.out.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0])
        writer.writeheader()
        writer.writerows(summaries)
    primary = {name: [r for r in summaries if r["dataset"] == name]
               for name in ("cora", "ogbn_arxiv_20k", "openalex_20k")}
    frozen_gate = sum(all(r["frozen_passes_5pp"] for r in rows) for rows in primary.values()) >= 2
    stable_gate = sum(all(r["stable_passes_5pp"] for r in rows) for rows in primary.values()) >= 2
    print("PRIMARY GATE", "frozen", frozen_gate, "stable", stable_gate, "output", args.out)


if __name__ == "__main__":
    main()
