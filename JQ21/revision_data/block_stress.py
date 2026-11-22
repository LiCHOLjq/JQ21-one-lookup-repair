"""Paired replay of the archived block-size study with native-size calibration."""

import argparse
import csv
from functools import lru_cache
from hashlib import file_digest
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time

for thread_var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[thread_var] = '1'

import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp, xlogy


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ORIGINAL = WORKSPACE / 'research_direction_20261119' / 'experiments'
DISCOVERY = WORKSPACE / 'research_discovery_20261115'
DATASETS = ('cora', 'ogbn_arxiv_20k', 'openalex_20k', 'amazon_computers')
SIZES = (4, 8, 12, 24)
SEED = 191126
PAIRWISE = 'native_pair_24'
HALVES = 'native_halves_24'
FIXED = 'archived_b6'
NATIVE = 'native_gibbs'
SINKHORN_ITERS = 200
BOOTSTRAP = 4000


@lru_cache(None)
def transitions(n):
    subset = np.arange(1 << n)
    cardinality = np.fromiter((int(m).bit_count() for m in subset), dtype=int, count=len(subset))
    bits = [1 << j for j in range(n)]
    table = []
    for k in range(n):
        layer = subset[cardinality == k]
        empty = [layer[(layer & bit) == 0] for bit in bits]
        table.append([(before, before | bit) for before, bit in zip(empty, bits)])
    return table


def reduced_scores(s):
    """Reweight rows/columns so an optimal assignment has all zero scores."""
    rows, pi = linear_sum_assignment(-s)
    arc = s[rows, pi, None] - s[:, pi]
    # A virtual source with zero outgoing costs supplies feasible potentials.
    h = np.zeros(len(s))
    for _ in range(len(s) - 1):
        h = np.minimum(h, (h[:, None] + arc).min(axis=0))
    reduced = np.empty_like(s)
    costs = arc + h[:, None] - h[None, :]
    if costs.min() < -1e-11:
        raise ArithmeticError('Invalid assignment reweighting')
    reduced[:, pi] = -np.maximum(costs, 0)
    return reduced


def permanent(s, temperatures, marginals=False):
    """Positive subset DP. Centering by an optimal assignment ensures Z >= 1."""
    n = len(s)
    reduced = reduced_scores(s)
    weights = np.exp(reduced[:, :, None] / temperatures)
    forward = np.zeros((1 << n, len(temperatures)))
    forward[0] = 1
    steps = transitions(n)
    for row, layer in enumerate(steps):
        for col, (before, after) in enumerate(layer):
            if len(before):
                forward[after] += forward[before] * weights[row, col]
    z = forward[-1]
    if not np.logical_and(np.isfinite(z), z >= 1).all():
        raise ArithmeticError('Invalid normalized permanent')
    if not marginals:
        return reduced, np.log(z)
    if len(temperatures) != 1:
        raise ValueError('Marginals require a single temperature')
    weight = weights[:, :, 0]
    backward = np.zeros(1 << n)
    backward[-1] = 1
    p = np.zeros_like(reduced)
    for row in reversed(range(n)):
        for col, (before, after) in enumerate(steps[row]):
            if len(before):
                totals = backward[after] * weight[row, col]
                backward[before] += totals
                p[row, col] = forward[before, 0] @ totals / z[0]
    if not (np.allclose(p.sum(0), 1, atol=1e-11, rtol=0) and np.allclose(p.sum(1), 1, atol=1e-11, rtol=0)):
        raise ArithmeticError('Inconsistent marginal sums')
    return reduced, np.log(z), p


def check_permanent():
    rng = np.random.default_rng(421996)
    checks = []
    for n in (2, 4, 6, 8):
        perms = np.array(list(itertools.permutations(range(n))))
        rows = np.arange(n)
        for kind in ('continuous', 'tied', 'all_tied', 'sharp'):
            if kind == 'continuous':
                s = rng.uniform(-1, 1, size=(n, n))
            elif kind == 'tied':
                s = rng.integers(-5, 6, size=(n, n)).astype(float)
            elif kind == 'all_tied':
                s = np.zeros((n, n))
            else:
                s = rng.uniform(-400, -100, size=(n, n)) + 700 * np.eye(n)
            map_score = s[rows, linear_sum_assignment(-s)[1]].sum()
            joint = s[rows, perms].sum(axis=1) - map_score
            truth = rng.permutation(n)
            for t in (0.008, 0.065, 0.3, 1.5):
                reduced, logz, p = permanent(s, np.array([t]), marginals=True)
                logz_ref = float(logsumexp(joint / t))
                weighted = np.exp(joint / t - logz_ref)
                p_ref = np.array([np.bincount(perms[:, i], weights=weighted, minlength=n) for i in rows])
                nll = float(logz[0] - reduced[rows, truth].sum() / t)
                nll_ref = logz_ref - (s[rows, truth].sum() - map_score) / t
                max_marginal_error = float(np.abs(p - p_ref).max())
                assert abs(logz[0] - logz_ref) < 1e-9 and abs(nll - nll_ref) < 1e-8 and max_marginal_error < 1e-9
                checks.append(dict(size=n, kind=kind, temperature=t, max_marginal_error=max_marginal_error,
                                   logz_error=float(abs(logz[0] - logz_ref)), nll_error=float(abs(nll - nll_ref))))
    return checks


def native_block_losses(s, truth, temperatures, objective):
    realigned = s[:, truth]
    if objective == PAIRWISE:
        row, col = np.triu_indices(len(s), 1)
        diag = np.diag(realigned)
        swap_gain = (realigned[row, col] + realigned[col, row]) - (diag[row] + diag[col])
        return np.logaddexp(0, swap_gain[:, None] / temperatures).mean(axis=0)
    if objective == HALVES:
        half = len(s) // 2
        # The calibration row order is uniformly shuffled before forming blocks.
        return (native_block_losses(realigned[:half, :half], np.arange(half), temperatures, NATIVE)
                + native_block_losses(realigned[half:, half:], np.arange(half), temperatures, NATIVE))
    reduced, logz = permanent(realigned, temperatures)
    return logz - np.trace(reduced) / temperatures


def native_fits(cal, size, contexts, features, temperatures):
    requests = list(make_blocks(cal, size, 591119 + size, None))
    objectives = (PAIRWISE, HALVES) if size == 24 else (NATIVE,)
    profiles = {method: {} for method in objectives}
    for scorer, context in contexts.items():
        totals = {name: np.zeros(len(temperatures)) for name in objectives}
        for ids, cols, truth in requests:
            s = context[ids] @ features[cols].T
            for name in objectives:
                totals[name] += native_block_losses(s, truth, temperatures, name)
        for name in objectives:
            profiles[name][scorer] = (totals[name] / len(requests)).tolist()

    result = {}
    for name in objectives:
        loss, scorer, index = min((v, n, j) for n, seq in profiles[name].items() for j, v in enumerate(seq))
        result[name] = dict(scorer=scorer, temperature=float(temperatures[index]), grid_index=index,
                            boundary=index in (0, len(temperatures) - 1), loss=loss,
                            calibration_blocks=len(requests), grid_losses=profiles[name])
    return result


def make_blocks(vertices, size, seed, limit):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(vertices)
    n = len(vertices) // size
    if limit is not None:
        n = min(n, limit)
    for block in range(n):
        ids = shuffled[block * size:(block + 1) * size]
        slots = rng.permutation(size)
        yield ids, ids[slots], np.argsort(slots)


def sinkhorn_entropy(s, temperature):
    p = s / temperature
    for _ in range(SINKHORN_ITERS):
        p = p - logsumexp(p, axis=1)[:, None]
        p = p - logsumexp(p, axis=0)
        if np.abs(np.exp(logsumexp(p, axis=1)) - 1).max() < 1e-7:
            break
    p -= logsumexp(p, axis=1)[:, None]
    marginal = np.exp(p)
    return -(marginal * p).sum(1), float(np.abs(marginal.sum(0) - 1).max())


def evaluate(s, truth, t, exchange, success):
    b = len(s)
    loss = np.sort(exchange(s), axis=1)[:, 1:]
    value = logsumexp(-loss / t, axis=1)
    approx, residual = sinkhorn_entropy(s, t)
    options = dict(value=value, sinkhorn_entropy=approx, forced_gap=-loss[:, 0])
    if b <= 12:
        _, _, p = permanent(s, np.array([t]), marginals=True)
        options['exact_entropy'] = -xlogy(p, p).sum(axis=1)
    ok = np.array([success(s, truth, row) for row in range(b)])
    result = dict(no_lookup=success(s, truth), random=float(ok.mean()), oracle=int(ok.max()),
                  sinkhorn_residual=residual, exact_entropy='', exact_entropy_ties='')
    for name, objective in options.items():
        tied = np.isclose(objective, objective.max(), rtol=1e-10, atol=1e-12)
        result[name] = float(ok[tied].mean())
        result[name + '_ties'] = int(tied.sum())
    return result


def load_graph(dataset, path):
    with np.load(path, allow_pickle=False) as data:
        if dataset != 'amazon_computers':
            return data['x'].astype(float), data['edges'].astype(int)
        def sparse_matrix(prefix):
            return sparse.csr_matrix((data[f'{prefix}_data'], data[f'{prefix}_indices'], data[f'{prefix}_indptr']),
                                     shape=tuple(data[f'{prefix}_shape']))
        return sparse_matrix('attr').toarray().astype('float32').astype(float), np.column_stack(sparse_matrix('adj').nonzero())


def sha256(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def rows_from_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def to_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, archive):
    keys = ('no_lookup', 'random', 'oracle', 'value', 'forced_gap', 'exact_entropy', 'sinkhorn_entropy')
    archived = {(r['dataset'], int(r['size'])): r for r in archive}
    for dataset in DATASETS:
        for size in SIZES:
            cell = [r for r in rows if (r['dataset'], r['size']) == (dataset, size)]
            baseline = [r for r in cell if r['fit'] == FIXED]
            positions = [(r['block'], r['vertices'], r['feature_slots']) for r in baseline]
            archived_row = archived[dataset, size]
            assert len(baseline) == int(archived_row['n'])
            for fit in dict.fromkeys(r['fit'] for r in cell):
                selected = [r for r in cell if r['fit'] == fit]
                assert [(r['block'], r['vertices'], r['feature_slots']) for r in selected] == positions
                out = dict(dataset=dataset, seed=SEED, size=size, fit=fit, blocks=len(selected))
                for key in keys:
                    out[key] = float(np.mean([r[key] for r in selected])) if selected[0][key] != '' else ''
                    out['archive_diff_' + key] = (out[key] - float(archived_row[key])) if fit == FIXED and key != 'exact_entropy' else ''
                if fit != FIXED:
                    delta = np.array([r['value'] - old['value'] for r, old in zip(selected, baseline)])
                    boot = np.random.default_rng(SEED + size).integers(0, len(delta), size=(BOOTSTRAP, len(delta)))
                    lo, hi = np.quantile(delta[boot].mean(1), [0.025, 0.975])
                    out.update(value_minus_archive=float(delta.mean()), ci_low=float(lo), ci_high=float(hi))
                else:
                    out.update(value_minus_archive='', ci_low='', ci_high='')
                yield out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', default=ORIGINAL, type=Path,
                        help='Source directory with the archived scripts and run metadata')
    parser.add_argument('--primary', default=DISCOVERY / 'data', type=Path)
    parser.add_argument('--openalex', default=DISCOVERY / 'extra_data' / 'openalex_20k.npz', type=Path)
    parser.add_argument('--amazon', type=Path, help='Amazon CSR input (default: experiments/public_data)')
    parser.add_argument('--out', default=HERE / 'block_stress', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.experiments.resolve()))
    from study import TEMP, exchange, predictions
    from scale_real import success

    start = time.monotonic()
    sources = (args.primary / 'cora.npz', args.primary / 'ogbn_arxiv_20k.npz', args.openalex,
               args.amazon or args.experiments / 'public_data' / 'amazon_original.npz')
    archived = {}
    for path in ('results/run.json', 'transfer_results/run.json'):
        for row in json.loads((args.experiments / path).read_text(encoding='utf-8'))['runs']:
            if row['seed'] == SEED:
                archived[row['dataset']] = row

    calibration = []
    records = []
    for dataset, source in zip(DATASETS, sources):
        x, edges = load_graph(dataset, source)
        order = np.random.default_rng(SEED).permutation(len(x))
        anchors, cal, test = np.split(order, [int(.55 * len(x)), int(.70 * len(x))])
        assert not (np.intersect1d(cal, test).size or np.intersect1d(anchors, test).size or np.intersect1d(anchors, cal).size)
        x, contexts, _ = predictions(x, edges, anchors)
        scorer = archived[dataset]['calibration_method']
        temperature = archived[dataset]['temperature']
        for size in SIZES:
            fits = {FIXED: dict(scorer=scorer, temperature=temperature)}
            fits.update(native_fits(cal, size, contexts, x, TEMP))
            for fit, detail in fits.items():
                calibration.append(dict(dataset=dataset, seed=SEED, size=size, fit=fit, **detail))
            for block, (ids, cols, truth) in enumerate(make_blocks(test, size, 391119 + size, 220)):
                for name, detail in fits.items():
                    s = contexts[detail['scorer']][ids] @ x[cols].T
                    score = evaluate(s, truth, detail['temperature'], exchange, success)
                    records.append(dict(dataset=dataset, seed=SEED, size=size, fit=name, block=block,
                                        vertices=' '.join(map(str, ids)), feature_slots=' '.join(map(str, cols)), **score))
            print(dataset, size, 'complete', flush=True)

    frozen_archive = args.experiments / 'block_sizes.csv'
    to_csv(args.out / 'blocks.csv', records)
    to_csv(args.out / 'summary.csv', list(summarize(records, rows_from_csv(frozen_archive))))
    oracle = check_permanent()
    to_csv(args.out / 'dp_oracle.csv', oracle)
    run = dict(seed=SEED, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
               system=platform.system(), seconds=time.monotonic()-start,
               candidates=TEMP.tolist(), bootstrap_resamples=BOOTSTRAP, sinkhorn_max_iterations=SINKHORN_ITERS,
               source_hashes={p.name: sha256(p) for p in [Path(__file__), frozen_archive, *sources,
                   args.experiments / 'study.py', args.experiments / 'scale_real.py']},
               primary_meta_sha256=sha256(args.experiments / 'results' / 'run.json'),
               transfer_meta_sha256=sha256(args.experiments / 'transfer_results' / 'run.json'),
               fits=calibration, oracle_checks=len(oracle))
    (args.out / 'run.json').write_text(json.dumps(run, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
