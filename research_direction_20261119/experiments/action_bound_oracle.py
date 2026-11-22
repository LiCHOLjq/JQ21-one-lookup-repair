"""Independent finite posterior check of mean(g) >= 2 max(g)/b.

The enumerated distributions are arbitrary, not score-factorized.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

SEED = 773319


def gains(permutations, probabilities):
    b = permutations.shape[1]
    map_index = int(np.argmax(probabilities))
    pi = permutations[map_index]
    forced = np.zeros((b,b))
    for i in range(b):
        np.maximum.at(forced[i],permutations[:,i],probabilities)
    return forced.sum(axis=1)-probabilities[map_index]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(exist_ok=False,parents=True)
    rng=np.random.default_rng(SEED);records=[]
    for b in range(2,9):
        perms=np.array(list(itertools.permutations(range(b))))
        for kind in ['dense','sparse','tied']:
            for replicate in range(20):
                if kind=='dense':p=rng.lognormal(0,3,len(perms))
                elif kind=='tied':p=rng.integers(0,5,len(perms)).astype(float)
                else:p=rng.exponential(size=len(perms))*(rng.random(len(perms))<.15)
                if not p.sum():p[0]=1
                p/=p.sum();g=gains(perms,p)
                slack=float(g.mean()-2*g.max()/b)
                assert slack>=-1e-13
                records.append(dict(b=b,kind=kind,replicate=replicate,
                    max_gain=float(g.max()),mean_gain=float(g.mean()),bound_slack=slack))
        p=np.zeros(len(perms));p[0]=1.0
        for j in range(1,b):
            candidate=np.arange(b);candidate[0],candidate[j]=j,0
            index=np.flatnonzero(np.all(perms==candidate,axis=1))[0]
            p[index]=rng.uniform(.01,.99)
        p/=p.sum();g=gains(perms,p)
        assert np.isclose(g.mean(),2*g.max()/b,rtol=1e-12,atol=1e-13)
        assert np.isclose(g[0],g[1:].sum(),rtol=1e-12,atol=1e-13)
        records.append(dict(b=b,kind='star_equality',replicate=0,
            max_gain=float(g.max()),mean_gain=float(g.mean()),
            bound_slack=float(g.mean()-2*g.max()/b)))
    with (args.out/'cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    meta=dict(seed=SEED,cases=len(records),minimum_slack=min(r['bound_slack'] for r in records),
        star_equalities=7,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Arbitrary finite permutation posterior checks supplement, not replace, the algebraic proof.')
    (args.out/'run.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))


if __name__=='__main__':main()
