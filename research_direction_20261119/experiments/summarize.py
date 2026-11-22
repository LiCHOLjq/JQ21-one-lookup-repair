"""Create paper tables from both server result bundles. No experiment is rerun."""
from pathlib import Path
import csv
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(p):
    with p.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

def save(name,rows):
    with (ROOT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def main():
    rng=np.random.default_rng(76119)
    stats=[];competitors=[];scale=[]
    for component in ['results','transfer_results']:
        folder=ROOT/component
        grouped=defaultdict(list)
        for r in read(folder/'blocks.csv'):
            grouped[(r['dataset'],r['seed'],r['regime'],r['control'])].append(r)
        for (ds,seed,reg,ctrl),rows in grouped.items():
            n=len(rows)
            keys=['unqueried','random','gap','entropy','value','oracle','rows_random','rows_value','same_query']
            a={k:np.array([float(r[k]) for r in rows]) for k in keys}
            out=dict(dataset=ds,seed=seed,regime=reg,control=ctrl,n=n)
            out.update({k:float(col.mean()) for k,col in a.items()})
            for k in ['random','entropy']:
                d=a['value']-a[k]
                means=np.concatenate([d[rng.integers(0,n,size=(100,n))].mean(axis=1)
                                      for _ in range(40)])
                low,high=np.quantile(means,[.025,.975])
                out.update({f'vs_{k}':d.mean(),f'vs_{k}_lo':low,f'vs_{k}_hi':high})
            stats.append(out)
        for file,keylist,target in [('secondary.csv',['dataset','seed','regime'],competitors),
                                    ('scale_real.csv',['dataset','size'],scale)]:
            grouped=defaultdict(list)
            for r in read(folder/file): grouped[tuple(r[k] for k in keylist)].append(r)
            for group,rows in grouped.items():
                out=dict(zip(keylist,group));out['n']=len(rows)
                for k in rows[0]:
                    if k not in keylist+['block']: out[k]=np.mean([float(r[k]) for r in rows])
                target.append(out)
    save('rates.csv',stats);save('strong_controls.csv',competitors);save('block_sizes.csv',scale)

if __name__=='__main__': main()
