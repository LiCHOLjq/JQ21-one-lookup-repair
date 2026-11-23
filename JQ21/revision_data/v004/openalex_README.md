# OpenAlex metadata ablation

The experiment crosses two graphs (the five-relation graph and its citation/shared-author subgraph) with two BGE input strings (the full cached text and title/abstract alone). The primary analysis uses the same BGE implementation and precision for both strings. The original cached features provide a separate numerical check. All test blocks, slots, anchor vertices, calibration blocks and scorer/temperature grids are matched within each split. The six-row block ordering follows the order of the archived uniform/aligned rows. Rates in the CSV files use a 0–1 scale.

## Recompute the six-row study

From the repository root, set the following paths in a POSIX shell. Three runs reproduce the summary files, per-block outcomes and scores. No download of raw papers or BGE weights is needed for these runs.

```sh
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
R=JQ21/revision_data/v004
E=research_direction_20261119/experiments
G=research_discovery_20261115/extra_data/openalex_20k.npz
python "$R/openalex_ablation.py" "$E" "$G" "$R/openalex_citation_author_edges.npy" replay
for V in matched_full title_abstract; do
  python "$R/openalex_ablation.py" "$E" "$G" "$R/openalex_citation_author_edges.npy" replay \
    --clean-vectors "$R/openalex_${V}_vectors.npz" --view "$V"
done
```

Each vector NPZ contains the `x` array for all 20,000 works in the same order as the frozen graph. Each score NPZ contains unnormalized six-by-six score matrices, true slot indices and fitted temperature, keyed by edge condition and split. NumPy/BLAS round-off can affect a first-maximum tie. On the experiment server, replay of all 12,000 scored blocks exactly recovered all five policy outcomes. An independent numerical environment reproduced every outcome except two entropy outcomes in the matched-full-text, citation/author condition (one block in each split). All outcomes in the joint ablation reproduced in both environments. The per-block CSV and frozen scores document the reported run. The paired bootstrap uses 4,000 resamples drawn within each graph condition and split.

## How the inputs were built

The full historical preprocessing export includes `paper_ids.csv`, the raw OpenAlex `works.jsonl`, and typed `edges.csv`. To rebuild the restricted graph directly, substitute the upstream typed edge CSV for the restricted NPY in the command above. The script verifies that the union of all five induced, symmetrized relations reproduces every edge of the original frozen 20,000-node graph before using only citations and shared authors.

To reconstruct the strings from that export, run the following. The supplied OpenAlex field/concept-name dictionary freezes the API lookup used for the suffix verification. None of the 20,000 works needed the fallback work API call.

```sh
python "$R/openalex_text.py" ORIGINAL/paper_ids.csv ORIGINAL/raw/openalex/works.jsonl \
  "$G" "$R/openalex_metadata_names.json" texts.jsonl.gz
```

For optional re-encoding, use the BAAI/bge-base-en-v1.5 checkpoint revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` and write that revision to `commit.txt` in the downloaded model folder. The encoding script uses Transformers 4.57.1 with PyTorch 2.7.1 (CPU), CLS pooling, a 512-token cap, bfloat16 evaluation, and float32 L2 normalization. Both encoding audits give the checkpoint digest, vector archive digest and full-text spot check. The supplied arrays avoid dependencies on processor and library precision. [BAAI checkpoint](https://huggingface.co/BAAI/bge-base-en-v1.5/tree/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a)

```sh
for V in matched_full title_abstract; do
  python "$R/openalex_encode.py" LOCAL_MODEL_FOLDER texts.jsonl.gz "$G" encoded --view "$V"
done
```

The data products are numeric derivatives of the original OpenAlex database. The release contains neither paper abstracts nor encoder weights. The original label-seeded vertex sample is common to all conditions.
