"""Repackage existing public graph data without reusing old queries or methods."""
from pathlib import Path
import hashlib,json,datetime,collections
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("input_directory", type=Path, help="Parent of cora and ogbn_arxiv_full preprocessing directories")
args=parser.parse_args()
BASE=Path(__file__).resolve().parent
OUT=BASE/'data'
OUT.mkdir(exist_ok=True)
SRC=args.input_directory
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
manifest=[]
for original,name,limit in [('cora','cora',None),('ogbn_arxiv_full','ogbn_arxiv_20k',20000)]:
    src=SRC/original
    v=np.load(src/'vectors.npy');y=np.load(src/'labels.npy');g=np.load(src/'graph_cache.npz')
    # Do not load the object-typed rel_names array. Numeric CSR arrays suffice.
    indptr=g['u_indptr'];indices=g['u_indices'];n=len(v)
    selected=np.arange(n,dtype=np.int32)
    if limit:
        rng=np.random.default_rng(20261115)
        roots=[]
        for label in np.unique(y):
            pool=np.flatnonzero(y==label);roots.extend(rng.choice(pool,min(3,len(pool)),replace=False).tolist())
        seen=set(roots); queue=collections.deque(roots)
        while queue and len(seen)<limit:
            a=queue.popleft()
            for b in indices[indptr[a]:indptr[a+1]]:
                b=int(b)
                if b not in seen:
                    seen.add(b);queue.append(b)
                    if len(seen)==limit:break
        selected=np.array(sorted(seen),dtype=np.int32)
    lookup=np.full(n,-1,dtype=np.int32);lookup[selected]=np.arange(len(selected))
    def edges_from(prefix):
        ptr=g[prefix+'_indptr'];ix=g[prefix+'_indices']; parts=[]
        for a in selected:
            bs=ix[ptr[a]:ptr[a+1]];bs=bs[lookup[bs]>=0]
            if len(bs):parts.append(np.column_stack((np.full(len(bs),lookup[a]),lookup[bs])))
        return np.unique(np.concatenate(parts).astype(np.int32),axis=0)
    xraw=v[selected].astype(np.float32)
    if original=='cora':x=normalize(TruncatedSVD(n_components=64,random_state=0).fit_transform(xraw)).astype(np.float32)
    else:x=normalize(xraw).astype(np.float32)
    dest=OUT/(name+'.npz')
    np.savez_compressed(dest,x=x,x_original=xraw,labels=y[selected].astype(str),ids=selected.astype(str),edges=edges_from('u'),directed_edges=edges_from('r0'))
    meta=json.loads((src/'metadata.json').read_text())
    item={'dataset':name,'prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_directory':str(src),'source_metadata':meta,'source_files_sha256':{f:digest(src/f) for f in ['vectors.npy','labels.npy','graph_cache.npz','metadata.json']},'output_sha256':digest(dest),'n_nodes':len(selected),'n_edges':len(edges_from('u')),'dimension':x.shape[1],'preprocessing':'Cora: existing features + SVD64(random_state=0) + L2 normalization; OGBN-Arxiv: existing 128D features + L2 normalization','selection':'all nodes' if limit is None else 'BFS induced 20000-node subset seeded by up to 3 nodes per class, RNG20261115','scope':'Existing public data only; no prior queries, algorithms or experiment results reused. Original source provenance retained; upstream raw bytes were not redownloaded in this run.'}
    manifest.append(item); print(json.dumps({k:item[k] for k in ['dataset','n_nodes','n_edges','dimension','output_sha256']}),flush=True)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
