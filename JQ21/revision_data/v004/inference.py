"""Paired policy-family tests and shuffled six-row controls for the v004 revision."""
import argparse
import csv
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.stats import binomtest

DATA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA))
from analyze_six_row import read_csv, replay, write_csv

DRAWS = 199999


def sign_test(delta, seed):
    """Two-sided sign-flip test on block-level differences, including ties."""
    delta = np.round(np.asarray(delta, dtype=float), 12)
    nonzero = delta[delta != 0]
    if not len(nonzero):
        return 1.0, "identical"
    magnitudes, counts = np.unique(np.abs(nonzero), return_counts=True)
    if len(magnitudes) == 1:
        return binomtest(np.count_nonzero(nonzero > 0), len(nonzero), 0.5).pvalue, "exact sign flip"
    threshold = abs(nonzero.sum()) - 1e-9
    if len(nonzero) <= 18:
        sums = np.array([0.0])
        for v in np.abs(nonzero):
            sums = np.concatenate((sums + v, sums - v))
        return np.mean(abs(sums) >= threshold), "exact sign flip"
    # Equal magnitudes share one binomial draw, which exactly represents their
    # independent random signs; no block weights or distinct magnitudes are lost.
    rng = np.random.default_rng(seed)
    permuted = np.zeros(DRAWS)
    for v, n in zip(magnitudes, counts):
        permuted += (2 * rng.binomial(n, .5, DRAWS) - n) * v
    return (1 + np.count_nonzero(abs(permuted) >= threshold)) / (1 + DRAWS), "Monte Carlo sign flip"


def add_contrast(out, b, conv, dataset, split, endpoint, policy, value, other):
    value, other = np.array(value), np.array(other)
    delta = value - other
    label = f"{b}/{conv}/{dataset}/{split}/{endpoint}/{policy}"
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    p, method = sign_test(delta, seed)
    if np.isin(value, [0, 1]).all() and np.isin(other, [0, 1]).all():
        method = "exact McNemar"
    out.append(dict(block_size=b, convention=conv, dataset=dataset, seed=split,
                    endpoint=endpoint, policy=policy, n=len(delta), rate_value=value.mean(),
                    rate_other=other.mean(), delta=delta.mean(), n_discordant=int((abs(delta)>1e-12).sum()),
                    test=method, draws=DRAWS if method=="Monte Carlo sign flip" else 0,
                    p_raw=float(p), p_holm=0.0))


def holm(rows):
    families = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("block_size", "convention", "endpoint", "policy"))
        families[key].append(row)
    for group in families.values():
        assert len(group) == 8
        maximum = 0
        for rank, r in enumerate(sorted(group, key=lambda i:i["p_raw"])):
            maximum = min(1, max(maximum, (8-rank)*r["p_raw"]))
            r["p_holm"] = maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA.parents[1])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    cells, _ = replay(args.root / "research_direction_20261119" / "experiments",
                      args.root / "research_discovery_20261115")
    tests, controls = [], []
    for (dataset, seed, regime, condition), blocks in cells.items():
        if condition == "aligned" and regime != "uniform":
            continue
        for conv, template in blocks[0][0].items():
            for index, endpoint in enumerate(["exact","row"]):
                for policy in template:
                    if template[policy][index] is None:
                        continue
                    other = [block[0][conv][policy][index] for block in blocks]
                    if condition == "shuffled":
                        if index == 0:
                            controls.append(dict(dataset=dataset, seed=seed, regime=regime, convention=conv,
                                                 policy=policy, n=len(other), rate=np.mean(other)))
                    elif policy != "value":
                        value = [x[0][conv]["value"][index] for x in blocks]
                        add_contrast(tests, 6, conv, dataset, seed, endpoint, policy, value, other)
    twelve = defaultdict(list)
    for row in read_csv(DATA / "twelve_policy_blocks.csv"):
        twelve[row["dataset"],row["seed"]].append(row)
    for (dataset, seed), rows in twelve.items():
        for endpoint in ("exact", "row"):
            prefix = endpoint + "_"
            for field in rows[0]:
                if field.startswith(prefix) and field != prefix+"value":
                    add_contrast(tests,12,"action_and_decoder_tie_average",dataset, seed, endpoint,
                                 field[len(prefix):],[float(b[prefix+"value"]) for b in rows],
                                 [float(b[field]) for b in rows])
    holm(tests)
    # Preserve numerical comparability with the archived rates used for the article.
    reference = {}
    for name in ["six_policy_comparisons.csv", "twelve_policy_comparisons.csv"]:
        for r in read_csv(DATA/name):
            if r["condition"] == "aligned" and r["regime"] == "uniform":
                reference[tuple(r[k] for k in ["block_size","convention","dataset","seed","endpoint","policy"])] = float(r["delta_value"])
    for r in tests:
        key = tuple(str(r[k]) for k in ["block_size","convention","dataset","seed","endpoint","policy"])
        assert abs(r["delta"]-reference[key]) < 1e-9, (key,r["delta"],reference[key])
    args.out.mkdir(exist_ok=True,parents=True)
    write_csv(args.out/"inference_tests.csv", tests)
    write_csv(args.out/"inference_shuffled.csv",controls)
    print("wrote",len(tests),"tests and",len(controls),"control rates")


if __name__ == "__main__":
    main()

