"""Reanalyse archived shuffles, selected settings and OpenAlex feature widths.

Run from any directory in the workspace or the extracted supplement.
"""
import csv
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
EXPERIMENTS = WORK / "research_direction_20261119" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
import study


def save(name, records):
    with (HERE / name).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def calibration_table():
    settings = []
    for path in ("results/run.json", "transfer_results/run.json",
                 "b12_confirmation/accelerated_results/run.json"):
        for run in json.loads((EXPERIMENTS / path).read_text())["runs"]:
            settings.append({key: run[key] for key in
                             ("dataset", "seed", "calibration_method", "temperature")})
    save("calibration_settings.csv", settings)


def shuffled_controls(blocks):
    grouped = {}
    for block in blocks:
        if block["control"] == "shuffled":
            key = (block["dataset"], block["seed"], block["regime"])
            grouped.setdefault(key, []).append(int(block["value"]))
    results = []
    for (dataset, seed, regime), hits in grouped.items():
        test = binomtest(sum(hits), len(hits), 1 / 120)
        ci = test.proportion_ci(method="exact")
        results.append(dict(dataset=dataset, seed=seed, regime=regime, n=len(hits),
                            successes=sum(hits), rate=sum(hits)/len(hits),
                            ci_low=ci.low, ci_high=ci.high, p_two_sided=test.pvalue))
    save("shuffled_controls.csv", results)


def openalex_widths(blocks):
    aligned = [b for b in blocks if b["dataset"] == "openalex_20k"
               and b["regime"] == "uniform" and b["control"] == "aligned"]
    with np.load(WORK / "research_discovery_20261115/extra_data/openalex_20k.npz") as cache:
        full = cache["x"].astype(float)
        edges = cache["edges"]
    results = []
    for seed in (191126, 201126):
        frozen = [b for b in aligned if int(b["seed"]) == seed]
        for width in (64, 128, full.shape[1]):
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(full))
            anchors, cal, _ = np.split(order, [int(.55*len(order)), int(.70*len(order))])
            # Match the primary stream up to calibration, including its 768-D direction.
            direction = rng.normal(size=full.shape[1])[:width]
            x, contexts, _ = study.predictions(full[:, :width], edges, anchors)
            model, temp, _, _ = study.calibrate(cal, contexts["centroid"], x, contexts,
                                                direction, rng)
            keys = ("unqueried", "random", "entropy", "value")
            evaluated = []
            for block in frozen:
                ids = np.fromstring(block["vertices"], sep=" ", dtype=int)
                slots = np.fromstring(block["feature_slots"], sep=" ", dtype=int)
                truth = (ids[:, None] == slots[None, :]).argmax(axis=1)
                values = study.evaluate(contexts[model][ids] @ x[slots].T, truth, temp)
                evaluated.append([values[key] for key in keys])
            averages = dict(zip(keys, np.mean(evaluated, axis=0)))
            if width == full.shape[1]:
                for key in ("value", "random", "unqueried"):
                    assert np.isclose(averages[key], np.mean([float(b[key]) for b in frozen]))
            results.append(dict(seed=seed, dimensions=width, blocks=len(frozen),
                                scorer=model, temperature=temp, **averages))
    save("openalex_widths.csv", results)


if __name__ == "__main__":
    records = []
    for source in ("results/blocks.csv", "transfer_results/blocks.csv"):
        with (EXPERIMENTS / source).open(newline="") as stream:
            records.extend(csv.DictReader(stream))
    calibration_table()
    shuffled_controls(records)
    openalex_widths(records)
