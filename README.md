# One-lookup verification for graph–feature joins

Research code and numerical data for *The Value of Choosing Which Record to Verify in Graph–Feature Joins*.

Starting from a score matrix on a known block of vertices and a shuffled batch of corresponding feature records, each policy chooses one vertex to verify. The primary outcome is correct recovery of the entire bijection after its true slot is revealed.

## Reproduce figures, tables and the paired comparison

Use Python 3.11 or later with NumPy, SciPy and Matplotlib. The requirements file lists the versions used for the archived revision. Run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python JQ21/plot_figures.py
python JQ21/revision_data/render_policy_tables.py
python JQ21/revision_data/six_entropy_holm.py
```

The plotting script writes nine PDFs to `JQ21/figures/`. The table generator writes LaTeX tabular material to the JQ21 directory. The Holm test script reproduces the discordant counts and two-sided exact McNemar p-values for the eight six-row comparisons of whole-block value with exact entropy, with Holm adjustment across the eight cells. These commands use included numerical outcomes and work offline after installing dependencies.

## Figure numbering in the Information Sciences manuscript

The four main figures use the files named `setting`, `gain_bound`, `primary_gain`, and `exact_entropy_confirmation` in order. Supplementary Figures S1–S5 use `exchange_paths`, `regime_controls`, `block_sizes`, `size_calibration`, and `kernel_timing`. The table generator emits study results in the original LaTeX layout. The journal manuscript and supplement use the same data with separate numbering.

## Repeat graph-level analyses

The numeric feature and edge caches for the ogbn-arxiv and OpenAlex subsets are included. Fetch Cora and Amazon Computers directly from their upstream sources and verify their digests before running the graph-level analyses. If network access is restricted, pass the original LINQS tarball and the pinned Amazon sparse file using `--cora-tgz FILE --amazon-npz FILE`.

```sh
python restore_inputs.py
python JQ21/revision_data/context_checks.py
python JQ21/revision_data/analyze_six_row.py
python JQ21/revision_data/block_stress.py
```

The twelve-row confirmation archive includes the per-block score matrices, truth, policy data and model configuration. Build the C++17 shared library to rerun the exact twelve-row comparison on these frozen inputs. Supply an absolute library path via `--kernel` if using a different build location.

```sh
c++ -O3 -std=c++17 -shared -fPIC JQ21/revision_data/twelve_policy_kernel.cpp -o JQ21/revision_data/twelve_policy_kernel.so
python JQ21/revision_data/twelve_policy_comparisons.py --kernel JQ21/revision_data/twelve_policy_kernel.so
```

On Windows, build the DLL with an x64 Visual Studio developer shell using the command in the Python driver's docstring. Rerunning an analysis updates its output files. Deterministic completion choices can differ across numerical libraries on tied scores. The archive comparison for the stress test is recorded in `JQ21/revision_data/block_stress/archive_replay_audit.json`.

## Data and provenance

| Source | Released material and preparation |
| --- | --- |
| Cora | The repository supplies the fitted, normalized 64-coordinate SVD result used by the study. The restore script reads the 2,708 records from the original LINQS download, rebuilds the normalized bag-of-words and edge arrays, and combines these with the frozen projection to reproduce the exact original study-cache SHA-256. The frozen projection avoids platform-dependent differences when refitting randomized SVD. Original tarball SHA-256 `0d4ed463d1627bb7f3e8420effe8f5545fd492ae8f88dab44ce86cee7b26d7e8`. [LINQS source tarball](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) |
| ogbn-arxiv | The 20,000-node induced subset includes normalized features and symmetrized edges. Original dataset by the Open Graph Benchmark team (Hu et al., 2020), based on Microsoft Academic Graph, available under the Open Data Commons Attribution License. This study sampled up to three roots per class and performed breadth-first selection with seed 20261115. [Original dataset](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) · [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/) |
| OpenAlex | The induced 20,000-work derivative supplies numerical embeddings, local IDs, numeric labels and edges. An earlier OpenAlex Works API query for *database systems* returned 80,797 articles. Cached BAAI/bge-base-en-v1.5 vectors encode the title, abstract, field/topic and venue fields. The graph combines citation, shared author, field, venue and year relations. The project manifest records the filter, source hashes, model name and seed 20261116. The shared feature matrix uses 768 coordinates. The upstream database is distributed as CC0. [OpenAlex data information](https://help.openalex.org/data/how-its-built/) · [Encoder](https://huggingface.co/BAAI/bge-base-en-v1.5) |
| Amazon Computers | The restore script retrieves the original sparse graph from commit `1e72912a0810cdf27ae54fd589a3b43358a2b161` of Shchur et al.'s GNN benchmark and checks SHA-256 `736ba1d9fd85eac2da83a5ad25c463f04b0031a6ca01563f90a7099d9e8ffb2c`. [Original data](https://github.com/shchur/gnn-benchmark/blob/1e72912a0810cdf27ae54fd589a3b43358a2b161/data/npz/amazon_electronics_computers.npz) |

Dataset terms apply to their respective upstream materials. The authors have not designated a separate redistribution licence for the study code.

## Directory map

- `JQ21/revision_data/` contains analysis drivers, the portable twelve-row kernel, policy comparisons, posterior diagnostics, calibration choices, overlap checks, dimension sensitivity and stress-test results.
- `JQ21/figure_data/` contains frozen aggregate inputs. Rates and contrasts in CSVs are fractions unless their column names specify another unit. For six-row results, distinguish the original first-maximum convention from tie-averaged supplemental comparisons.
- `research_direction_20261119/experiments/` contains the original study code, protocols, frozen block/slot orderings and outcomes, synthetic kernel benchmarks, and twelve-row confirmation.
- `research_discovery_20261115/` contains the input manifests, numeric derivatives, and the subset-construction code. Re-running the subset-construction scripts requires the earlier graph exports specified in those scripts. The frozen derivatives and restore script supply the inputs needed for the paper analyses.
