"""Blinded six-slot attachment recovery. See PROTOCOL.md. Python + NumPy + SciPy."""
import argparse
import csv
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

B = 6
ROWS = np.arange(B)
PERMS = np.array(list(itertools.permutations(range(B))))
TEMP = np.geomspace(.008, 1.5, 16)


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def exchange(s):
    """Return cost of the best forced completion relative to the MAP optimum."""
    rows, pi = linear_sum_assignment(-s)
    arc = s[rows, pi][:, None] - s[:, pi]
    dist = arc.copy()
    for k in rows:
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    loss = np.empty_like(arc)
    loss[:, pi] = np.maximum(0., arc + dist.T)
    return loss


def blocks(nodes, context, regime, rng, direction):
    if regime == 'uniform':
        order = rng.permutation(nodes)
    else:
        sorted_ids = nodes[np.argsort(context[nodes] @ direction, kind='stable')]
        order = np.concatenate([rng.permutation(sorted_ids[i:i+30])
                                for i in range(0, len(sorted_ids), 30)])
    for start in range(0, len(order)-B+1, B):
        ids = order[start:start+B]
        slots = rng.permutation(B)
        yield ids, ids[slots], np.argsort(slots)


def predictions(x, edges, anchors):
    n, d = x.shape
    src, dst = edges.T
    adj = sparse.csr_matrix((np.ones(len(src)), (src,dst)), shape=(n,n))
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1
    sub = adj[:, anchors]
    counts = np.asarray(sub.sum(1)).ravel()
    centered = norm(norm(x)-norm(x[anchors]).mean(0))
    h = (sub @ centered[anchors]) / np.maximum(1, counts[:,None])
    contexts = {'centroid': norm(h)}
    cov = h[anchors].T @ h[anchors]
    target = h[anchors].T @ centered[anchors]
    for strength in [.1, 1., 10.]:
        penalty = strength * max(np.trace(cov)/d, 1e-8)
        weights = np.linalg.solve(cov + penalty*np.eye(d), target)
        contexts['ridge_'+str(strength)] = norm(h @ weights)
    return centered, contexts, counts


def calibrate(cal, h, x, contexts, direction, rng):
    requests = list(blocks(cal,h,'uniform',rng,direction))
    losses = {}
    for name, pred in contexts.items():
        totals = np.zeros(len(TEMP))
        for ids, cols, truth in requests:
            s = pred[ids] @ x[cols].T
            joint = s[ROWS, PERMS].sum(1)
            true_score = s[ROWS, truth].sum()
            totals += logsumexp((joint[:, None]-true_score)/TEMP, axis=0)
        losses[name] = (totals/len(requests)).tolist()
    val, name, index = min((v,n,j) for n,vs in losses.items() for j,v in enumerate(vs))
    return name, TEMP[index], val, losses


def evaluate(s, truth, temperature):
    joint = s[ROWS, PERMS].sum(1)
    mass = np.exp((joint-joint.max())/temperature)
    mass /= mass.sum()
    prob = np.stack([np.bincount(PERMS[:, i], weights=mass, minlength=B) for i in ROWS])
    entropy = -(prob * np.log(np.maximum(prob,1e-300))).sum(1)
    sorted_s = np.sort(s,axis=1)
    gap = sorted_s[:,-1]-sorted_s[:,-2]
    values = np.exp(-exchange(s)/temperature).sum(1)
    policies = {'entropy': int(np.argmax(entropy)), 'gap': int(np.argmin(gap)),
                'value': int(np.argmax(values))}
    # Independent slot randomization makes lexicographic ties label-blind.
    blind = PERMS[np.argmax(joint)]
    match = np.zeros((B,B),bool)
    for i in ROWS:
        allowed = np.flatnonzero(PERMS[:,i]==truth[i])
        decoded = PERMS[allowed[np.argmax(joint[allowed])]]
        match[i] = decoded == truth
    ok = match.all(1)
    result = {'unqueried':int(np.all(blind==truth)), 'rows_unqueried':np.mean(blind==truth),
              'random':ok.mean(),'rows_random':match.mean(),'oracle':int(ok.any()),
              'same_query':int(policies['value']==policies['entropy'])}
    for key, i in policies.items():
        result[key] = int(ok[i])
        result['rows_'+key] = match[i].mean()
        result['asked_'+key] = i
    return result


def run(file, seed, writer, meta):
    data = np.load(file, allow_pickle=False)
    x = data['x'].astype(float)
    edges = data['edges'].astype(int)
    assert x.ndim==2 and edges.shape[1]==2 and np.isfinite(x).all()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    anchors, cal, test = np.split(order, [int(.55*len(x)), int(.70*len(x))])
    x, contexts, counts = predictions(x,edges,anchors)
    h = contexts['centroid']
    direction = rng.normal(size=x.shape[1])
    name, t, val, losses = calibrate(cal,h,x,contexts,direction,rng)
    meta.append(dict(dataset=file.stem,seed=seed,n=len(x),d=x.shape[1],
                     cached_edge_rows=len(edges),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                     anchor_n=len(anchors),cal_n=len(cal),test_n=len(test),
                     zero_anchor_fraction=float((counts[test]==0).mean()),
                     calibration_method=name,temperature=t,nll=val,
                     grid=TEMP.tolist(),calibration_nll=losses))
    for regime in ['uniform','banded']:
        for j,(ids,cols,truth) in enumerate(blocks(test,h,regime,rng,direction)):
            s = contexts[name][ids] @ x[cols].T
            for control, score in [('aligned',s),('shuffled',s[rng.permutation(B)])]:
                row=dict(dataset=file.stem,seed=seed,regime=regime,control=control,block=j,
                         vertices=' '.join(map(str,ids)),feature_slots=' '.join(map(str,cols)),
                         zero_anchor_rows=int((counts[ids]==0).sum()))
                row.update(evaluate(score, truth, t))
                writer(row)


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument('--data', type=Path, required=True)
    arg.add_argument('--out', type=Path, required=True)
    args=arg.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    meta=[]
    start=time.monotonic()
    with (args.out/'blocks.csv').open('x',newline='') as stream:
        table=None
        def write(row):
            nonlocal table
            if table is None:
                table=csv.DictWriter(stream,fieldnames=row)
                table.writeheader()
            table.writerow(row)
        for name in ['cora','ogbn_arxiv_20k','openalex_20k']:
            for seed in [191126,201126]:
                run(args.data/(name+'.npz'),seed,write,meta)
                stream.flush()
                print(name,seed,'finished',flush=True)
    (args.out/'run.json').write_text(json.dumps(dict(runs=meta,seconds=time.monotonic()-start,
        python=platform.python_version(),platform=platform.system(),numpy=np.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':
    main()
