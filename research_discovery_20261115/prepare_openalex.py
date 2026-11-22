"""Prepare a new real-feature graph dataset; never reuse old workload outputs."""
from pathlib import Path
import collections, datetime, hashlib, json
import numpy as np

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("input_directory", type=Path, help="Directory with the 80,797-work export and precomputed vectors")
args=parser.parse_args()
ROOT=Path(__file__).resolve().parent
SRC=args.input_directory
OUT=ROOT/'extra_data'; OUT.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
v=np.load(SRC/'vectors.npy',mmap_mode='r'); y=np.load(SRC/'labels.npy')
z=np.load(SRC/'graph_cache.npz'); ptr=z['u_indptr']; ix=z['u_indices']; n=len(v)
rng=np.random.default_rng(20261116)
roots=[]
for label in np.unique(y):
    pool=np.flatnonzero(y==label); roots.extend(rng.choice(pool,min(3,len(pool)),replace=False).tolist())
seen=set(roots); queue=collections.deque(roots)
while queue and len(seen)<20000:
    a=queue.popleft()
    for b in ix[ptr[a]:ptr[a+1]]:
        b=int(b)
        if b not in seen:
            seen.add(b); queue.append(b)
            if len(seen)==20000:break
sel=np.array(sorted(seen),dtype=np.int32)
lookup=np.full(n,-1,dtype=np.int32); lookup[sel]=np.arange(len(sel))
parts=[]
for a in sel:
    bs=lookup[ix[ptr[a]:ptr[a+1]]];bs=bs[bs>=0]
    if len(bs):parts.append(np.column_stack((np.full(len(bs),lookup[a]),bs)))
edges=np.unique(np.concatenate(parts).astype(np.int32),axis=0)
x=np.array(v[sel],dtype=np.float32);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
dest=OUT/'openalex_20k.npz'
np.savez_compressed(dest,x=x,labels=y[sel].astype(str),ids=sel.astype(str),edges=edges)
meta=dict(dataset='openalex_20k',prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),n=len(sel),dim=x.shape[1],edge_rows=len(edges),source_directory=str(SRC),source_metadata=json.loads((SRC/'metadata.json').read_text()),embedding_metadata=json.loads((SRC/'openalex_fetch_metadata.json').read_text()),selection='BFS induced 20000 nodes; up to 3 seed nodes per label; RNG20261116',graph_semantics='Cached union of cites/same_author/same_field/same_venue/same_year relations, symmetrized. NOT pure directed citations.',embedding='Cached BAAI/bge-base-en-v1.5 768D title/abstract/field/topic/venue embeddings according to inherited metadata; L2 normalized again. Encoder not rerun or independently authenticated.',source_hashes={f:sha(SRC/f) for f in ['vectors.npy','labels.npy','graph_cache.npz','metadata.json','openalex_fetch_metadata.json']},output_sha256=sha(dest),source_code_sha256=sha(Path(__file__)),limitation='Source cache reused read-only; no old queries, methods, results used. Original API records and model version were not redownloaded in this run.')
(OUT/'openalex_20k_manifest.json').write_text(json.dumps(meta,indent=2))
print(json.dumps({k:meta[k] for k in ['dataset','n','dim','edge_rows','output_sha256']}))
