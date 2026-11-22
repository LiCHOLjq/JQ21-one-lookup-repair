"""Additional literature-based policies on the frozen test requests, with uniform tie breaking."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from study import B, ROWS, PERMS, predictions

def score_policies(s, truth, t):
    joint=s[ROWS,PERMS].sum(1)
    p=np.exp((joint-joint.max())/t)
    p/=p.sum()
    pair_ids=PERMS.T[:,None,:]*B+PERMS.T[None,:,:]
    marg=np.array([np.bincount(PERMS[:,i],weights=p,minlength=B) for i in ROWS])
    rowmax=np.empty((B,B))
    query_ok=np.empty((B,B),bool)
    for i in ROWS:
        for j in ROWS:
            cand=np.flatnonzero(PERMS[:,i]==j)
            rowmax[i,j]=p[cand].max()
        cand=np.flatnonzero(PERMS[:,i]==truth[i])
        query_ok[i]=PERMS[cand[np.argmax(joint[cand])]]==truth
    # Malmi et al. Eq. 5, with exact marginals. Including i adds the same 1 to all.
    look=np.zeros(B)
    for i in ROWS:
        for k in ROWS:
            joint_table=np.bincount(pair_ids[i,k],weights=p,minlength=B*B).reshape(B,B)
            look[i]+=joint_table.max(axis=1).sum()
    objectives={'whole_value':rowmax.sum(1),'expected_rows':look,
                'least_confident':-marg.max(1), 'forced_runner_up':np.sort(rowmax,axis=1)[:,-2],
                'marginal_entropy':-(marg*np.log(np.maximum(marg,1e-300))).sum(1)}
    attr={}
    for name,vals in objectives.items():
        top=np.isclose(vals,vals.max(),rtol=1e-11,atol=1e-15)
        attr[name]=query_ok.all(1)[top].mean()
        attr['rows_'+name]=query_ok[top].mean()
    gain=rowmax.sum(1)-p.max()
    attr['best_model_gain']=max(gain)
    attr['random_model_gain']=gain.mean()
    assert gain.mean()+1e-12 >= 2*max(gain)/B
    return attr

def main():
    parser=argparse.ArgumentParser()
    for flag in ['data','results','out']: parser.add_argument('--'+flag,required=True,type=Path)
    args=parser.parse_args()
    meta=json.loads((args.results/'run.json').read_text())['runs']
    with (args.results/'blocks.csv').open() as f:
        blocks=[r for r in csv.DictReader(f) if r['control']=='aligned']
    with args.out.open('x',newline='') as f:
        writer=None
        for m in meta:
            rng=np.random.default_rng(m['seed'])
            z=np.load(args.data/(m['dataset']+'.npz'))
            x=z['x'].astype(float)
            anchors=rng.permutation(len(x))[:int(.55*len(x))]
            x,ctx,_=predictions(x,z['edges'].astype(int),anchors)
            for r in blocks:
                if (r['dataset'],int(r['seed']))!=(m['dataset'],m['seed']): continue
                ids=np.fromstring(r['vertices'],sep=' ',dtype=int)
                cols=np.fromstring(r['feature_slots'],sep=' ',dtype=int)
                truth=np.argmax(ids[:,None]==cols[None,:],axis=1)
                s=ctx[m['calibration_method']][ids]@x[cols].T
                v=score_policies(s,truth,m['temperature'])
                out={k:r[k] for k in ['dataset','seed','regime','block']}
                out.update(v)
                if writer is None:
                    writer=csv.DictWriter(f,fieldnames=out)
                    writer.writeheader()
                writer.writerow(out)
            f.flush()
            print(m['dataset'],m['seed'],'done',flush=True)
if __name__=='__main__': main()
