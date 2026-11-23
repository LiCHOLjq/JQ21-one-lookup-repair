"""Six-row matched OpenAlex ablation; use the original study.py and frozen slots."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np


def save_csv(path, rows):
    with path.open('w', encoding='utf8',newline='') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_edges(source, selected, frozen):
    remap = {int(v):i for i,v in enumerate(selected)}
    all_edges, clean = set(), set()
    with source.open(newline='') as f:
        for row in csv.DictReader(f):
            u,v = remap.get(int(row['src'])),remap.get(int(row['dst']))
            if u is None or v is None or u == v:
                continue
            all_edges.update(((u,v),(v,u)))
            if row['relation'] in ('cites','same_author'):
                clean.update(((u,v),(v,u)))
    assert np.array_equal(np.array(sorted(all_edges)),frozen)
    return np.array(sorted(clean),dtype=np.int32)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('experiment',type=Path,help='directory containing study.py and results/blocks.csv')
    a.add_argument('cache',type=Path)
    a.add_argument('typed_edges',type=Path,help='original typed edge CSV, or supplied restricted edge NPY')
    a.add_argument('out',type=Path)
    a.add_argument('--clean-vectors',type=Path,help='re-encoded vectors in frozen row order')
    a.add_argument('--view',choices=['title_abstract','matched_full'],default='title_abstract')
    args = a.parse_args()
    sys.path.insert(0,str(args.experiment.resolve()))
    import study
    args.out.mkdir(exist_ok=True,parents=True)
    data = np.load(args.cache)
    full = data['x']
    edges = data['edges']
    if args.typed_edges.suffix=='.npy':
        restricted=np.load(args.typed_edges)
        assert restricted.shape==(148354,2) and restricted.min()>=0 and restricted.max()<len(full)
    else:
        restricted = read_edges(args.typed_edges,data['ids'],edges)
    with (args.experiment/'results/blocks.csv').open(newline='') as f:
        blocks = [b for b in csv.DictReader(f) if b['dataset']=='openalex_20k'
                  and b['regime']=='uniform' and b['control']=='aligned']
    views = [('original',full)]
    if args.clean_vectors:
        vec=np.load(args.clean_vectors)
        views = [(args.view,vec['x'] if args.clean_vectors.suffix=='.npz' else vec)]
    for view, array in views:
        assert array.shape == full.shape and np.isfinite(array).all()
        output, summary, frozen_scores = [],[],{}
        for channel, adjacency in [('all',edges),('citation_author',restricted)]:
            for seed in [191126,201126]:
                rng=np.random.default_rng(seed)
                order=rng.permutation(len(full))
                anchor,cal,_=np.split(order,[11000,14000])
                direction=rng.normal(size=768)
                x, contexts,counts=study.predictions(array.astype(float),adjacency,anchor)
                method,temp,_,_=study.calibrate(cal,contexts['centroid'],x,contexts,direction,rng)
                outcomes, matrices, truths=[],[],[]
                for b in blocks:
                    if int(b['seed'])!=seed:
                        continue
                    ids=np.fromstring(b['vertices'],dtype=int,sep=' ')
                    slots=np.fromstring(b['feature_slots'],dtype=int,sep=' ')
                    truth=(ids[:,None]==slots[None,:]).argmax(1)
                    score=contexts[method][ids]@x[slots].T
                    result=study.evaluate(score,truth,temp)
                    matrices.append(score)
                    truths.append(truth)
                    outcomes.append({k:result[k] for k in ('unqueried','random','gap','entropy','value')})
                    if channel=='all' and view=='original':
                        assert all(np.isclose(float(b[k]),result[k]) for k in ('value','random','unqueried'))
                assert len(outcomes)==1000
                prefix=channel+'_'+str(seed)
                frozen_scores[prefix+'_scores']=np.array(matrices)
                frozen_scores[prefix+'_truth']=np.array(truths,dtype=np.int8)
                frozen_scores[prefix+'_temperature']=np.asarray(temp)
                for i,row in enumerate(outcomes):
                    output.append(dict(features=view,edges=channel,seed=seed,block=i,**row))
                record=dict(features=view,edges=channel,seed=seed,n=len(outcomes),
                            oriented_edges=len(adjacency),scorer=method,temperature=temp,
                            zero_anchor_neighbors=int((counts[order[14000:]]==0).sum()))
                for k in outcomes[0]:
                    record[k]=np.mean([r[k] for r in outcomes])
                vals=np.array([[r[k] for k in ('value','random','entropy')] for r in outcomes])
                boot=np.random.default_rng(20261123+seed)
                ix=boot.integers(0,len(outcomes),size=(4000,len(outcomes)))
                for j,k in [(1,'random'),(2,'entropy')]:
                    diff=vals[:,0]-vals[:,j]
                    record['value_minus_'+k]=diff.mean()
                    record[k+'_ci_low'],record[k+'_ci_high']=np.quantile(diff[ix].mean(1),[.025,.975])
                summary.append(record)
                print(view,channel,seed,record['unqueried'],record['value'],flush=True)
        save_csv(args.out/('openalex_'+view+'_blocks.csv'),output)
        save_csv(args.out/('openalex_'+view+'.csv'),summary)
        np.savez_compressed(args.out/('openalex_'+view+'_scores.npz'),**frozen_scores)
    clean_path=args.out/'openalex_citation_author_edges.npy'
    np.save(clean_path,restricted)


if __name__=='__main__':
    main()
