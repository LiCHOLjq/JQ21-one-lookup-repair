"""Independent assignment-sensitivity oracle and single-thread scaling benchmark.

Seed 73319. The two-cycle and best-two-assignment utilities are approximations;
all-pairs exchange, forced Hungarian, and exhaustive enumeration are exact.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

SEED = 73319


def map_assignment(s):
    rows, pi = linear_sum_assignment(s, maximize=True)
    return pi, s[rows, pi].sum()


def exchange_loss(s):
    n = len(s)
    pi, optimum = map_assignment(s)
    c = s[np.arange(n), pi][:, None] - s[:, pi]
    d = c.copy()
    for k in range(n):
        d = np.minimum(d, d[:, k, None] + d[None, k, :])
    result = np.empty_like(s)
    result[:, pi] = np.maximum(0.0, c + d.T)
    return result


def forced_hungarian_loss(s):
    n = len(s)
    _, optimum = map_assignment(s)
    loss = np.empty_like(s)
    for i in range(n):
        without_row = np.delete(s, i, axis=0)
        for j in range(n):
            remaining = np.delete(without_row, j, axis=1)
            a, b = linear_sum_assignment(remaining, maximize=True)
            loss[i, j] = optimum - s[i, j] - remaining[a, b].sum()
    return np.maximum(loss, 0.0)


def exhaustive_loss(s):
    n = len(s)
    permutations = np.array(list(itertools.permutations(range(n))), dtype=int)
    scores = s[np.arange(n), permutations].sum(axis=1)
    constrained = np.full((n, n), -np.inf)
    for i in range(n):
        np.maximum.at(constrained[i], permutations[:, i], scores)
    return scores.max() - constrained


def utilities(loss, temperature):
    return np.exp(-loss / temperature).sum(axis=1)


def exchange_utility(s, temperature):
    return utilities(exchange_loss(s), temperature)


def forced_utility(s, temperature):
    return utilities(forced_hungarian_loss(s), temperature)


def pair_swap_utility(s, temperature):
    pi, _ = map_assignment(s)
    c = s[np.arange(len(s)), pi][:, None] - s[:, pi]
    # Feasible completion for i->pi(j): transpose i and j in the MAP.
    return utilities(np.maximum(0.0, c + c.T), temperature)


def top_two_assignment_utility(s, temperature):
    n = len(s)
    pi, optimum = map_assignment(s)
    second, second_score = None, -np.inf
    # Every different matching excludes at least one MAP edge. The maximum
    # over these n subproblems is therefore the exact second-best matching.
    for i in range(n):
        masked = s.copy()
        masked[i, pi[i]] = -np.inf
        candidate, score = map_assignment(masked)
        if score > second_score:
            second, second_score = candidate, score
    # Approximate all-action utility using only these two matchings.
    return 1.0 + (pi != second) * np.exp((second_score-optimum)/temperature)


def row_top_two_gap(s, temperature):
    # Row-local policy baseline, not an estimate of whole-block utility.
    top = np.partition(s, -2, axis=1)[:, -2:]
    return -np.abs(top[:, 1] - top[:, 0])


def generate(rng, n, kind):
    if kind == 'signed_gaussian':
        s = rng.normal(size=(n, n)) - 2.0
    elif kind == 'ties':
        s = rng.integers(-2, 3, size=(n, n)).astype(float)
    elif kind == 'all_tied':
        s = np.zeros((n, n))
    elif kind == 'heterogeneous_margins':
        s = rng.normal(scale=0.03, size=(n, n))
        s[np.diag_indices(n)] += np.geomspace(1e-4, 3.0, n)
        s += rng.normal(size=(n, 1)) + rng.normal(size=(1, n))
    else:
        raise ValueError(kind)
    return np.ascontiguousarray(s[:, rng.permutation(n)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)
    arrays, checks, timings, comparisons = {}, [], [], []
    kinds = ['signed_gaussian', 'ties', 'all_tied', 'heterogeneous_margins']
    max_error, max_utility_error = 0.0, 0.0
    for n in [2, 3, 4, 5, 6, 7, 8]:
        for kind in kinds:
            for replicate in range(5):
                key = f'correctness_{n}_{kind}_{replicate}'
                s = generate(rng, n, kind)
                arrays[key] = s
                exchange = exchange_loss(s)
                brute = exhaustive_loss(s)
                forced = forced_hungarian_loss(s)
                error = max(np.abs(exchange-brute).max(), np.abs(exchange-forced).max())
                utility_error = max(np.abs(utilities(exchange,t)-utilities(brute,t)).max()
                                    for t in [.01, .2, 1., 5.])
                assert error < 1e-9, (key, float(error))
                assert utility_error < 1e-9, (key, float(utility_error))
                checks.append(dict(case=key,n=n,kind=kind,replicate=replicate,
                                   max_loss_error=float(error),max_utility_error=float(utility_error)))
                max_error = max(max_error, error)
                max_utility_error = max(max_utility_error, utility_error)
    print('exhaustive correctness passed', len(checks), flush=True)
    methods = {'exchange_exact': exchange_utility, 'forced_hungarian_exact': forced_utility,
               'pair_swap_approx': pair_swap_utility,
               'top_two_assignment_approx': top_two_assignment_utility,
               'row_top_two_gap_policy': row_top_two_gap}
    for n in [4, 8, 16, 32, 64, 128, 256]:
        for kind in kinds:
            key = f'scaling_{n}_{kind}'
            s = generate(rng,n,kind)
            arrays[key] = s
            exact = exchange_utility(s, .2)
            for name, fn in methods.items():
                if name == 'forced_hungarian_exact' and n > 128:
                    continue
                value = fn(s,.2)  # warmup and comparison, outside timing
                if name == 'forced_hungarian_exact':
                    assert np.max(np.abs(value-exact)) < 1e-8
                if name in ('pair_swap_approx','top_two_assignment_approx'):
                    assert np.all(value <= exact + 1e-8)
                chosen = int(np.argmax(value))
                comparisons.append(dict(n=n,kind=kind,method=name,
                    max_utility_error=float(np.max(np.abs(value-exact)))
                        if name!='row_top_two_gap_policy' else None,
                    decision_optimal=bool(np.isclose(exact[chosen],exact.max(),rtol=1e-10,atol=1e-10)),
                    relative_action_regret=float((exact.max()-exact[chosen])/exact.max())))
                start = time.perf_counter(); fn(s,.2); warm_time = time.perf_counter()-start
                batch = max(1,min(100,int(.015/max(warm_time,1e-9))))
                for repeat in range(5):
                    start = time.perf_counter()
                    for _ in range(batch):fn(s,.2)
                    elapsed = time.perf_counter()-start
                    timings.append(dict(n=n,kind=kind,method=name,repeat=repeat,
                                        iterations=batch,seconds_per_call=elapsed/batch))
            print('scaling finished', n, kind, flush=True)
    for filename, records in [('correctness.csv',checks),('timings.csv',timings),
                              ('approximation_comparisons.csv',comparisons)]:
        with (args.out/filename).open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
    np.savez_compressed(args.out/'score_matrices.npz',**arrays)
    meta = dict(seed=SEED,correctness_cases=len(checks),max_loss_error=float(max_error),
                max_utility_error=float(max_utility_error),numpy=np.__version__,scipy=scipy.__version__,
                python=platform.python_version(),system=platform.system(),
                threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                estimand='sum_j exp(-(MAP score - best score forced to row i,column j)/T)',
                temperature=.2,
                note='Approximate methods are labeled; their errors and action regret are reported. No I/O or score construction included in kernel timings.')
    (args.out/'run.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)


if __name__=='__main__':
    main()
