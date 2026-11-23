"""Replay the large-block cells using text-only embeddings and citation/shared-author edges."""
from pathlib import Path
import argparse
import csv

from sequential import HERE, EXP, format_row, large, make_blocks, np, predictions, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", required=True, type=Path, help="Embedding NPY, or NPZ with x array")
    parser.add_argument("--edges", default=HERE/"openalex_citation_author_edges.npy", type=Path)
    parser.add_argument("--fits", default=HERE/"openalex_title_abstract.csv", type=Path)
    args = parser.parse_args()
    if args.vectors.suffix == ".npz":
        with np.load(args.vectors, allow_pickle=False) as data:
            features = data["x"].astype(float)
    else:
        features = np.load(args.vectors, allow_pickle=False).astype(float)
    edges = np.load(args.edges, allow_pickle=False)
    assert features.shape==(20_000,768) and np.isfinite(features).all()
    assert edges.shape==(148354,2) and 0<=edges.min() and edges.max()<len(features)
    with args.fits.open(newline="") as stream:
        fits = {int(r["seed"]): r for r in csv.DictReader(stream)
                if r["edges"]=="citation_author" and r["features"]=="title_abstract"}
    assert set(fits) == {191126,201126}
    blocks = []
    summary = []
    for seed, fit in fits.items():
        order = np.random.default_rng(seed).permutation(len(features))
        anchors, _, test = np.split(order, [11_000,14_000])
        x,contexts,_ = predictions(features, edges, anchors)
        context = contexts[fit["scorer"]]
        temp = float(fit["temperature"])
        # Verify the transported fit by independent reproduction of no-lookup
        # success on every frozen primary six-row block for this seed.
        hits = []
        with (EXP/"results/blocks.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                if (row["dataset"],int(row["seed"]),row["regime"],row["control"]) != ("openalex_20k",seed,"uniform","aligned"):
                    continue
                ids,slots=(np.fromstring(row[k],sep=" ",dtype=int) for k in ("vertices","feature_slots"))
                truth = (ids[:,None]==slots).argmax(axis=1)
                from study import PERMS
                joint = (context[ids]@x[slots].T)[np.arange(6),PERMS].sum(axis=1)
                hits.append(np.array_equal(PERMS[np.argmax(joint)],truth))
        assert len(hits)==1000 and np.isclose(np.mean(hits),float(fit["unqueried"]),rtol=0,atol=1e-12)
        for size in (24,32):
            group=[]
            for i,(v,c,truth) in enumerate(make_blocks(test,size,391119+size,220)):
                result=large(context[v]@x[c].T,truth,temp)
                group.append(format_row("openalex_title_abstract_citation_author",seed,size,1,i,result))
            assert len(group)==min(220,len(test)//size)
            blocks.extend(group)
            row=dict(seed=seed,size=size,n=len(group),scorer=fit["scorer"],temperature=temp)
            for key in group[0]:
                if key.endswith(("_exact","_rows")): row[key]=float(np.mean([r[key] for r in group]))
            summary.append(row)
    write(HERE/"sequential_clean_blocks.csv",blocks)
    write(HERE/"sequential_clean_summary.csv",summary)


if __name__=="__main__": main()
