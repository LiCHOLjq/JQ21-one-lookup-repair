"""Fixed-fold follow-up: adaptive lookup budgets and 24/32-row exact/partial recovery."""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
EXP = WORK / "research_direction_20261119/experiments"
DISC = WORK / "research_discovery_20261115"
sys.path[:0] = [str(EXP), str(HERE.parent)]
from study import PERMS, exchange, predictions
from block_stress import make_blocks, load_graph

B = 6
SUBSETS = [list(itertools.combinations(range(B), k)) for k in range(4)]


@lru_cache(None)
def lattice():
    # A state's fibers group all permutations giving the same answers.
    groups, key_to_fiber, parent_views = {}, {}, {}
    for level in SUBSETS:
        for rows in level:
            fibers = {}
            for i, perm in enumerate(PERMS):
                key = tuple(perm[j] for j in rows)
                fibers.setdefault(key, []).append(i)
            key_to_fiber[rows] = {key: i for i, key in enumerate(fibers)}
            groups[rows] = np.array(list(fibers.values()))
            for j in rows:
                parent = tuple(i for i in rows if i != j)
                owners = [key_to_fiber[parent][tuple(a for i, a in zip(rows, key) if i != j)]
                          for key in fibers]
                parent_views[parent, rows] = np.argsort(owners, kind="stable").reshape(
                    len(groups[parent]), -1)
    return groups, key_to_fiber, parent_views


def six(score, truth, temp, verify=False):
    groups, keys, views = lattice()
    joint = score[np.arange(B), PERMS].sum(axis=1)
    weight = np.exp((joint-joint.max())/temp)
    maximum = {rows: weight[g].max(axis=1) for rows, g in groups.items()}
    truth_ids = {r: keys[r][tuple(truth[j] for j in r)] for r in groups}
    hit = {}
    for r, g in groups.items():
        fiber = g[truth_ids[r]]
        decode = PERMS[fiber[np.argmax(joint[fiber])]]
        hit[r] = np.array([np.array_equal(decode, truth), np.mean(decode == truth)])
    # Report the expected posterior improvement in unnormalized units. The MAP
    # branch cancels: this avoids subtracting nearly equal success probabilities.
    gains, actions = {}, {}
    for h in (1, 2, 3):
        for size in range(4-h):
            for r in SUBSETS[size]:
                remaining = [j for j in range(B) if j not in r]
                branches = []
                for j in remaining:
                    child = tuple(sorted(r+(j,)))
                    view = views[r, child]
                    g = np.sort(maximum[child][view], axis=1)[:, :-1].sum(axis=1)
                    if h > 1:
                        g = g + gains[child, h-1][view].sum(axis=1)
                    branches.append(g)
                branch = np.array(branches)
                gains[r, h] = branch.max(axis=0)
                actions[r, h] = remaining[np.argmax(branch[:, truth_ids[r]])]
    if verify:
        def recurse(indices, remaining, horizon):
            if not horizon:
                return weight[indices].max()
            candidates = []
            for j in remaining:
                other = tuple(i for i in remaining if i != j)
                answers = PERMS[indices, j]
                candidates.append(sum(recurse(indices[answers == answer], other, horizon-1)
                                      for answer in np.unique(answers)))
            return max(candidates)
        for k in (1, 2, 3):
            reference = recurse(np.arange(len(PERMS)), tuple(range(B)), k)
            assert np.isclose(weight.max()+gains[(), k][0],reference,rtol=1e-12,atol=1e-12)
    def choose(k, lookahead):
        r = ()
        for step in range(k):
            j = actions[r, k-step if lookahead else 1]
            r = tuple(sorted(r+(j,)))
        return hit[r]
    out = []
    for k in (1, 2, 3):
        out.append(dict(budget=k, none=hit[()], uniform=np.mean([hit[s] for s in SUBSETS[k]],axis=0),
                        greedy=choose(k,False), optimal=choose(k,True)))
    return out


def large(score, truth, temp):
    n = len(truth)
    rows = np.arange(n)
    def outcome(j=None):
        keep = rows if j is None else rows[rows != j]
        cols = rows if j is None else rows[rows != truth[j]]
        r, c = linear_sum_assignment(-score[np.ix_(keep,cols)])
        count = int(j is not None) + np.count_nonzero(cols[c] == truth[keep[r]])
        return np.array([int(count == n), count/n])
    actions = np.array([outcome(j) for j in rows])
    obj = logsumexp(-np.sort(exchange(score),axis=1)[:,1:]/temp,axis=1)
    tie = np.isclose(obj, obj.max(), rtol=1e-10, atol=1e-12)
    return dict(none=outcome(),uniform=actions.mean(axis=0),value=actions[tie].mean(axis=0),
                hindsight=actions.max(axis=0))


def format_row(dataset,seed,n,k,block,states):
    rec = dict(dataset=dataset,seed=seed,size=n,budget=k,block=block)
    for name, a in states.items():
        rec[name+"_exact"], rec[name+"_rows"] = map(float,a)
    return rec


def run_cell(args):
    data, seed = args
    folder = "transfer_results" if data=="amazon_computers" else "results"
    if data in ("cora","ogbn_arxiv_20k"):
        cache = DISC/"data"/(data+".npz")
    elif data=="openalex_20k":
        cache = DISC/"extra_data"/(data+".npz")
    else:
        cache = EXP/"public_data/amazon_original.npz"
    fit=next(r for r in json.loads((EXP/folder/"run.json").read_text())["runs"]
             if r["dataset"]==data and r["seed"]==seed)
    x, edges = load_graph(data,cache)
    a,_,test=np.split(np.random.default_rng(seed).permutation(len(x)),[int(.55*len(x)),int(.70*len(x))])
    x,contexts,_=predictions(x,edges,a)
    prediction=contexts[fit["calibration_method"]]
    temp=fit["temperature"]
    rec=[]
    discord=[0,0,0]
    with (EXP/folder/"blocks.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            if not (row["dataset"]==data and int(row["seed"])==seed and
                    row["regime"]=="uniform" and row["control"]=="aligned"):
                continue
            v,c=(np.fromstring(row[f],dtype=int,sep=" ") for f in ("vertices","feature_slots"))
            truth=(v[:,None]==c).argmax(axis=1)
            rates=six(prediction[v]@x[c].T,truth,temp)
            for result in rates:
                k=result.pop("budget")
                rec.append(format_row(data,seed,B,k,int(row["block"]),result))
            discord[0] += abs(rec[-3]["none_exact"]-float(row["unqueried"]))>1e-9
            discord[1] += abs(rec[-3]["uniform_exact"]-float(row["random"]))>1e-9
            discord[2] += abs(rec[-3]["greedy_exact"]-float(row["value"]))>1e-9
    for size in (24,32):
        for block,(v,c,truth) in enumerate(make_blocks(test,size,391119+size,220)):
            rec.append(format_row(data,seed,size,1,block,large(prediction[v]@x[c].T,truth,temp)))
    print(data,seed,len(rec),"k=1 baseline discrepancies",discord,flush=True)
    return rec,dict(dataset=data,seed=seed,none=discord[0],uniform=discord[1],value=discord[2])


def write(path, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def checks():
    # Validate the whole state lattice and compare optimal model values against
    # an independent recursive decision tree for continuous, tied and flat scores.
    rng=np.random.default_rng(135)
    for score in [rng.normal(size=(6,6)),np.zeros((6,6)),rng.integers(-2,3,(6,6))]:
        truth=rng.permutation(6)
        result=six(score,truth,.7,verify=True)
        assert np.allclose(result[0]["greedy"],result[0]["optimal"])
        values=[large(score,truth,.7)[key] for key in ("none","uniform")]
        if len(np.unique(score))==36:
            assert np.allclose(values,[result[0]["none"],result[0]["uniform"]])
    groups,_,views=lattice()
    for (parent,child), ids in views.items():
        assert np.array_equal(np.sort(groups[child][ids].reshape(len(ids),-1),axis=1),groups[parent])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    start=time.monotonic()
    checks()
    cells=list(itertools.product(("cora","ogbn_arxiv_20k","openalex_20k","amazon_computers"),(191126,201126)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(run_cell,cells))
    rows=[r for batch,_ in results for r in batch]
    write(HERE/"sequential_blocks.csv",rows)
    summary=[]
    subsets={}
    for r in rows:
        key=tuple(r[k] for k in ("dataset","seed","size","budget"))
        subsets.setdefault(key,[]).append(r)
    for (data,seed,size,k),batch in subsets.items():
        rec=dict(dataset=data,seed=seed,size=size,budget=k,n=len(batch))
        for key in batch[0]:
            if key.endswith(("_exact","_rows")):
                rec[key]=float(np.mean([r[key] for r in batch]))
        summary.append(rec)
    write(HERE/"sequential_summary.csv",summary)
    info=dict(numpy=np.__version__,scipy=scipy.__version__,python=sys.version.split()[0],
              elapsed_seconds=time.monotonic()-start,workers=args.workers,checks=[row for _,row in results])
    (HERE/"sequential_run.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf8")


if __name__=="__main__": main()
