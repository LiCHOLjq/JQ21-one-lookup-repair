"""Encode the retained papers using original BGE CLS pooling and a 512-token limit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


def encode(texts, model, tokenizer, batch):
    tokens=tokenizer(texts,add_special_tokens=True,truncation=True,max_length=512,
                     padding=False,return_attention_mask=False)
    ids=tokens['input_ids']
    order=np.argsort([len(t) for t in ids],kind='stable')
    vectors=np.empty((len(texts),768),dtype=np.float32)
    start=monotonic()
    with torch.inference_mode():
        for i in range(0,len(order),batch):
            rows=order[i:i+batch]
            inputs=tokenizer.pad([{'input_ids':ids[j]} for j in rows],padding=True,
                                 pad_to_multiple_of=8,return_tensors='pt')
            output=model(**inputs).last_hidden_state[:,0].float()
            vectors[rows]=torch.nn.functional.normalize(output,dim=1).numpy()
            if i%(batch*50)==0:
                print(f'{i}/{len(texts)}, {monotonic()-start:.1f}s',flush=True)
    return vectors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path)
    parser.add_argument('texts',type=Path)
    parser.add_argument('cache',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--batch',type=int,default=24)
    parser.add_argument('--threads',type=int,default=20)
    parser.add_argument('--view',choices=['title_abstract','matched_full'],default='title_abstract')
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    with gzip.open(args.texts,'rt',encoding='utf8') as f:
        rows=[json.loads(l) for l in f]
    assert [r[0] for r in rows]==list(range(20000))
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModel.from_pretrained(args.model,local_files_only=True).eval().to(torch.bfloat16)
    spot=np.random.default_rng(20261123).choice(len(rows),size=128,replace=False)
    check=encode([rows[i][2] for i in spot],model,tokenizer,args.batch)
    frozen=np.load(args.cache)['x'][spot].astype(float)
    original_cos=np.sum(check*frozen,axis=1)/np.linalg.norm(frozen,axis=1)
    audit=dict(model='BAAI/bge-base-en-v1.5',revision=(args.model/'commit.txt').read_text().strip(),
               weight_sha256=hashlib.sha256((args.model/'model.safetensors').read_bytes()).hexdigest(),
               pooling='CLS and L2 normalization',max_tokens=512,cpu_dtype='bfloat16',
               torch=torch.__version__,transformers=transformers.__version__,spot_n=len(spot),
               spot_min_cosine=original_cos.min(),spot_mean_cosine=original_cos.mean(),seed=20261123)
    print(json.dumps(audit),flush=True)
    assert original_cos.mean()>.99 and original_cos.min()>.97
    column = 1 if args.view=='title_abstract' else 2
    output=encode([r[column] for r in rows],model,tokenizer,args.batch)
    args.out.mkdir(parents=True,exist_ok=True)
    vector_file=args.out/('openalex_'+args.view+'_vectors.npz')
    np.savez_compressed(vector_file,x=output)
    audit['vectors_sha256']=hashlib.sha256(vector_file.read_bytes()).hexdigest()
    audit['view']=args.view
    (args.out/('openalex_encoding_'+args.view+'.json')).write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
