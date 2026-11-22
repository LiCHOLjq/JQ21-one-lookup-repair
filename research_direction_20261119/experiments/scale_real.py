"""Exploratory block-size stress, with assignment-aware entropy approximation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from study import predictions, exchange

def policies(s,t):
    b=len(s)
    # exp(-loss/T) has unit value on a MAP edge; remove that constant
    # analytically using sorted losses so near-certain blocks stay well scaled.
    loss=np.sort(exchange(s),axis=1)[:,1:]
    value=logsumexp(-loss/t,axis=1)
    # Sinkhorn row / column normalizations for an entropic assignment,
    # an approximation to marginal uncertainty rather than the Gibbs marginals.
    logp=s/t
    for _ in range(200):
        logp=logp-logsumexp(logp,axis=1)[:,None]
        logp=logp-logsumexp(logp,axis=0)
        if np.max(np.abs(np.exp(logsumexp(logp,axis=1))-1))<1e-7: break
    logp-=logsumexp(logp,axis=1)[:,None]
    entropy=-(np.exp(logp)*logp).sum(1)
    return {'value':value, 'sinkhorn_entropy':entropy, 'forced_gap':-loss[:,0]}

def success(s,truth,asked=None):
    rows=np.arange(len(s))
    if asked is None:
        cols=rows
    else:
        cols=rows[rows!=truth[asked]]
        rows=rows[rows!=asked]
    _,assign=linear_sum_assignment(-s[np.ix_(rows,cols)])
    return int(np.array_equal(cols[assign], truth[rows]))

def main():
    ap=argparse.ArgumentParser()
    for name in ['data','meta','out']: ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    with args.out.open('x',newline='') as f:
        writer=None
        for m in json.loads(args.meta.read_text())['runs']:
            if m['seed']!=191126: continue
            data=np.load(args.data/(m['dataset']+'.npz'))
            x=data['x'].astype(float)
            order=np.random.default_rng(m['seed']).permutation(len(x))
            train=order[:int(.55*len(x))];test=order[int(.70*len(x)):]
            x,ctx,_=predictions(x,data['edges'],train)
            for b in [4,8,12,24]:
                rng=np.random.default_rng(391119+b)
                shuffled=rng.permutation(test)
                for j in range(min(len(test)//b,220)):
                    ids=shuffled[j*b:(j+1)*b]
                    perm=rng.permutation(b)
                    truth=np.argsort(perm)
                    s=ctx[m['calibration_method']][ids]@x[ids[perm]].T
                    options=policies(s,m['temperature'])
                    ok=np.array([success(s,truth,i) for i in range(b)])
                    out=dict(dataset=m['dataset'],size=b,block=j,no_lookup=success(s,truth),
                             random=ok.mean(),oracle=ok.max())
                    for k,vals in options.items():
                        out[k]=ok[np.isclose(vals,vals.max(),rtol=1e-10,atol=1e-12)].mean()
                    if writer is None:
                        writer=csv.DictWriter(f,fieldnames=out)
                        writer.writeheader()
                    writer.writerow(out)
            f.flush()
if __name__=='__main__': main()
