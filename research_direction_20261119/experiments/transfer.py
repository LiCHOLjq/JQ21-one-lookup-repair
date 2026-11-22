"""External replication on the original Amazon Computers sparse graph file."""
import csv
import json
from pathlib import Path
import argparse
import numpy as np
from scipy.sparse import csr_matrix
from study import run

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    raw=np.load(args.original,allow_pickle=False)
    def load(prefix):
        return csr_matrix((raw[prefix+'_data'],raw[prefix+'_indices'],raw[prefix+'_indptr']),
                          shape=tuple(raw[prefix+'_shape']))
    edges=np.column_stack(load('adj').nonzero())
    file=args.out/'amazon_computers.npz'
    np.savez_compressed(file,x=load('attr').toarray().astype('float32'),edges=edges)
    meta=[]
    with (args.out/'blocks.csv').open('x',newline='') as f:
        writer=None
        def emit(row):
            nonlocal writer
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
        for seed in [191126,201126]:
            run(file,seed,emit,meta)
    (args.out/'run.json').write_text(json.dumps({'runs':meta},indent=2))
if __name__=='__main__': main()
