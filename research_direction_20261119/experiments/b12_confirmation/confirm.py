"""Exact-marginal B=12 confirmation. Protocol is frozen in PROTOCOL.md."""
import argparse
import csv
import datetime
import hashlib
import itertools
import json
import os
import platform
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy
from scipy.special import logsumexp, xlogy

from study_frozen import predictions, calibrate, exchange

SCORE_TIE_ATOL = 1e-12
POLICIES = ['exact_entropy', 'sinkhorn_entropy', 'value', 'random', 'forced_gap']


@lru_cache(None)
def subset_structure(n):
    masks = np.arange(1 << n, dtype=np.int64)
    sizes = np.array([int(m).bit_count() for m in masks])
    bits = 1 << np.arange(n, dtype=np.int64)
    layers = [masks[sizes == k] for k in range(n+1)]
    return layers, bits


def permanent_and_map(s, temperature):
    """Forward/backward permanent and max/count semirings, all forced edges."""
    n = len(s)
    layers, bits = subset_structure(n)
    length = 1 << n
    a = s / temperature
    forward = np.full(length, -np.inf); forward[0] = 0.
    backward = np.full(length, -np.inf); backward[-1] = 0.
    best_f = np.full(length, -np.inf); best_f[0] = 0.
    best_b = np.full(length, -np.inf); best_b[-1] = 0.
    count_f = np.zeros(length, dtype=np.int64); count_f[0] = 1
    count_b = np.zeros(length, dtype=np.int64); count_b[-1] = 1
    for k in range(1, n+1):
        masks = layers[k]
        present = (masks[:, None] & bits) != 0
        previous = masks[:, None] ^ bits
        terms = np.where(present, forward[previous] + a[k-1], -np.inf)
        forward[masks] = logsumexp(terms, axis=1)
        terms = np.where(present, best_f[previous] + s[k-1], -np.inf)
        maximum = terms.max(axis=1)
        tied = present & np.isclose(terms, maximum[:, None], rtol=0, atol=SCORE_TIE_ATOL)
        best_f[masks] = maximum
        count_f[masks] = np.where(tied, count_f[previous], 0).sum(axis=1)
    for k in range(n-1, -1, -1):
        masks = layers[k]
        absent = (masks[:, None] & bits) == 0
        following = masks[:, None] | bits
        terms = np.where(absent, a[k] + backward[following], -np.inf)
        backward[masks] = logsumexp(terms, axis=1)
        terms = np.where(absent, s[k] + best_b[following], -np.inf)
        maximum = terms.max(axis=1)
        tied = absent & np.isclose(terms, maximum[:, None], rtol=0, atol=SCORE_TIE_ATOL)
        best_b[masks] = maximum
        count_b[masks] = np.where(tied, count_b[following], 0).sum(axis=1)
    logz = forward[-1]
    marginal = np.empty_like(s)
    forced_best = np.empty_like(s)
    forced_counts = np.empty(s.shape, dtype=np.int64)
    for i in range(n):
        masks = layers[i]
        absent = (masks[:, None] & bits) == 0
        following = masks[:, None] | bits
        terms = np.where(absent, forward[masks, None]+a[i]+backward[following], -np.inf)
        marginal[i] = np.exp(logsumexp(terms, axis=0)-logz)
        terms = np.where(absent, best_f[masks, None]+s[i]+best_b[following], -np.inf)
        maximum = terms.max(axis=0)
        tied = absent & np.isclose(terms, maximum[None, :], rtol=0, atol=SCORE_TIE_ATOL)
        forced_best[i] = maximum
        forced_counts[i] = np.where(tied, count_f[masks, None]*count_b[following], 0).sum(axis=0)
    assert np.max(np.abs(marginal.sum(1)-1)) < 1e-9
    assert np.max(np.abs(marginal.sum(0)-1)) < 1e-9
    assert np.all(forced_counts > 0)
    return marginal, logz, best_f[-1], int(count_f[-1]), forced_best, forced_counts


def brute_force(s, temperature):
    n = len(s)
    permutations = np.array(list(itertools.permutations(range(n))), dtype=np.int64)
    scores = s[np.arange(n), permutations].sum(1)
    logz = logsumexp(scores/temperature)
    p = np.exp(scores/temperature-logz)
    marginal = np.array([np.bincount(permutations[:, i], weights=p, minlength=n)
                         for i in range(n)])
    best = np.empty_like(s)
    counts = np.empty(s.shape, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            options = scores[permutations[:, i] == j]
            best[i, j] = options.max()
            counts[i, j] = np.isclose(options, options.max(), rtol=0, atol=SCORE_TIE_ATOL).sum()
    return marginal, logz, scores.max(), int(np.isclose(scores,scores.max(),rtol=0,atol=SCORE_TIE_ATOL).sum()), best, counts


def validate():
    rng = np.random.default_rng(912119)
    checks = []
    for n in range(2,9):
        for kind in ['signed', 'integer_ties', 'all_tied', 'heterogeneous']:
            for rep in range(2):
                if kind == 'signed': s = rng.normal(size=(n,n))-3
                elif kind == 'integer_ties': s = rng.integers(-2,3,(n,n)).astype(float)
                elif kind == 'all_tied': s = np.zeros((n,n))
                else:
                    s = rng.normal(scale=.02,size=(n,n))
                    s[np.diag_indices(n)] += np.geomspace(1e-4,2,n)
                    s += rng.normal(size=(n,1))
                s = s[:,rng.permutation(n)]
                for t in [.03, 1.2]:
                    dp = permanent_and_map(s,t); brute = brute_force(s,t)
                    errors = [float(np.max(np.abs(dp[k]-brute[k]))) for k in [0,1,2,4]]
                    assert max(errors) < 1e-9, (n,kind,errors)
                    assert dp[3] == brute[3]
                    assert np.array_equal(dp[5],brute[5]), (n,kind,dp[5],brute[5])
                    truth = rng.permutation(n); rows=np.arange(n)
                    truth_score=s[rows,truth].sum()
                    success_dp=np.isclose(truth_score,dp[4][rows,truth],rtol=0,atol=SCORE_TIE_ATOL)/dp[5][rows,truth]
                    success_brute=np.isclose(truth_score,brute[4][rows,truth],rtol=0,atol=SCORE_TIE_ATOL)/brute[5][rows,truth]
                    assert np.array_equal(success_dp,success_brute)
                    checks.append(dict(n=n,kind=kind,replicate=rep,temperature=t,
                        marginal_error=errors[0],logz_error=errors[1],map_error=errors[2],
                        forced_score_error=errors[3],counts_equal=True,decoder_outcomes_equal=True))
    return checks


def sinkhorn(s,t):
    logp=s/t
    residual=np.inf
    for iteration in range(10000):
        logp -= logsumexp(logp,axis=1)[:,None]
        logp -= logsumexp(logp,axis=0)
        residual=float(np.max(np.abs(np.exp(logsumexp(logp,axis=1))-1)))
        if residual<1e-10:break
    logp -= logsumexp(logp,axis=1)[:,None]
    p=np.exp(logp)
    residual=max(float(np.max(np.abs(p.sum(0)-1))),float(np.max(np.abs(p.sum(1)-1))))
    return p,iteration+1,residual


def tie_distribution(values):
    tied=np.isclose(values,values.max(),rtol=1e-10,atol=1e-12)
    return tied/tied.sum()


def evaluate(s,truth,t):
    n=len(s); rows=np.arange(n)
    marginal,logz,optimum,total_count,forced_best,counts=permanent_and_map(s,t)
    approximation,iterations,residual=sinkhorn(s,t)
    loss=exchange(s)
    forced_error=float(np.max(np.abs(loss-(optimum-forced_best))))
    assert forced_error<1e-8
    alternatives=np.sort(loss,axis=1)[:,1:]
    objectives={'exact_entropy':-xlogy(marginal,marginal).sum(1),
                'sinkhorn_entropy':-xlogy(approximation,approximation).sum(1),
                'value':logsumexp(-alternatives/t,axis=1),
                'random':np.zeros(n),'forced_gap':-alternatives[:,0]}
    choices=np.stack([tie_distribution(objectives[k]) for k in POLICIES])
    truth_score=s[rows,truth].sum()
    forced_count_truth=counts[rows,truth]
    ok=np.isclose(truth_score,forced_best[rows,truth],rtol=0,atol=SCORE_TIE_ATOL)/forced_count_truth
    out={name:float(choices[i]@ok) for i,name in enumerate(POLICIES)}
    out.update(no_lookup=float(np.isclose(truth_score,optimum,rtol=0,atol=SCORE_TIE_ATOL)/total_count),
        oracle=float(ok.max()),sinkhorn_iterations=iterations,sinkhorn_residual=residual,
        marginal_mean_absolute_error=float(np.abs(marginal-approximation).mean()),
        exact_sinkhorn_query_disagreement=float(1-choices[0]@choices[1]),
        value_exact_query_disagreement=float(1-choices[2]@choices[0]),
        decoder_tied_actions=int((forced_count_truth>1).sum()),max_decoder_ties=int(forced_count_truth.max()),
        no_lookup_decoder_ties=total_count,forced_score_validation_error=forced_error)
    for i,name in enumerate(POLICIES):out['tied_actions_'+name]=int(np.count_nonzero(choices[i]))
    raw=dict(score=s,truth=truth,true_marginals=marginal,sinkhorn_marginals=approximation,
        forced_loss=loss,forced_counts=counts,action_success=ok,policy_action_probabilities=choices,
        policy_objectives=np.stack([objectives[k] for k in POLICIES]))
    return out,raw


def write_csv(path,records):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)


def paired_summary(rows,seed):
    rng=np.random.default_rng(seed+440121)
    comparisons=[('value','exact_entropy'),('value','sinkhorn_entropy'),
                 ('exact_entropy','sinkhorn_entropy'),('value','random'),('value','forced_gap')]
    matrix=np.array([[r[a]-r[b] for a,b in comparisons] for r in rows])
    boot=[]
    for _ in range(10):
        ids=rng.integers(0,len(rows),(1000,len(rows)))
        boot.append(matrix[ids].mean(axis=1))
    ci=np.quantile(np.concatenate(boot),[.025,.975],axis=0)
    return [dict(dataset=rows[0]['dataset'],seed=seed,blocks=len(rows),policy=a,comparator=b,
                 policy_mean=float(np.mean([r[a] for r in rows])),
                 comparator_mean=float(np.mean([r[b] for r in rows])),
                 difference=float(matrix[:,i].mean()),ci_low=float(ci[0,i]),ci_high=float(ci[1,i]))
            for i,(a,b) in enumerate(comparisons)]


def run_graph(file,seed,out):
    data=np.load(file,allow_pickle=False)
    x=data['x'].astype(float); edges=data['edges'].astype(int)
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(x))
    anchors,cal,test=np.split(order,[int(.55*len(x)),int(.70*len(x))])
    x,contexts,counts=predictions(x,edges,anchors)
    h=contexts['centroid']; direction=rng.normal(size=x.shape[1])
    name,t,nll,grid=calibrate(cal,h,x,contexts,direction,rng)
    shuffled=rng.permutation(test)
    result=[];arrays={}
    for block in range(min(len(test)//12,220)):
        ids=shuffled[block*12:(block+1)*12]
        slots=rng.permutation(12);cols=ids[slots];truth=np.argsort(slots)
        s=contexts[name][ids]@x[cols].T
        values,raw=evaluate(s,truth,t)
        row=dict(dataset=file.stem,seed=seed,block=block,
                 vertices=' '.join(map(str,ids)),feature_slots=' '.join(map(str,cols)),
                 zero_anchor_rows=int((counts[ids]==0).sum()))
        row.update(values);result.append(row)
        for k,v in raw.items():arrays.setdefault(k,[]).append(v)
    tag=f'{file.stem}_{seed}'
    np.savez_compressed(out/(tag+'_raw.npz'),**{k:np.stack(v) for k,v in arrays.items()},
                        anchors=anchors,calibration_vertices=cal,test_vertices=test)
    meta=dict(dataset=file.stem,seed=seed,n=len(x),d=x.shape[1],edge_rows=len(edges),
        input_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),anchor_n=len(anchors),
        calibration_n=len(cal),test_n=len(test),blocks=len(result),calibration_method=name,
        temperature=float(t),calibration_nll=float(nll),calibration_grid_losses=grid,
        sinkhorn_max_residual=max(r['sinkhorn_residual'] for r in result),
        sinkhorn_nonconverged_blocks=sum(r['sinkhorn_residual']>1e-8 for r in result))
    return result,meta,paired_summary(result,seed)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--primary',type=Path,required=True)
    parser.add_argument('--amazon',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    checks=validate();write_csv(args.out/'oracle_checks.csv',checks)
    print('DP oracle passed',len(checks),flush=True)
    sources=[args.primary/(n+'.npz') for n in ['cora','ogbn_arxiv_20k','openalex_20k']]+[args.amazon]
    all_rows=[];runs=[];summaries=[]
    for file in sources:
        for seed in [119003,119004]:
            rows,meta,summary=run_graph(file,seed,args.out)
            all_rows.extend(rows);runs.append(meta);summaries.extend(summary)
            print(file.stem,seed,'finished',len(rows),flush=True)
    write_csv(args.out/'blocks.csv',all_rows);write_csv(args.out/'paired_summary.csv',summaries)
    meta=dict(start_utc=started,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seconds=time.monotonic()-start,runs=runs,policies=POLICIES,numpy=np.__version__,
        scipy=scipy.__version__,python=platform.python_version(),system=platform.system(),
        source_hashes={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
                       for n in ['confirm.py','study_frozen.py','PROTOCOL.md']},
        threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
        bootstrap_replicates=10000,oracle_cases=len(checks),
        note='New splits of the same four graphs; conditional paired intervals; injected 12-row faults, not real incidents.')
    (args.out/'run.json').write_text(json.dumps(meta,indent=2))
    print('confirmation complete',len(all_rows),flush=True)


if __name__=='__main__':main()
