# B=12 exact-entropy confirmation — frozen before this experiment's outcomes

This targeted confirmation follows an exploratory b=12 advantage against Sinkhorn entropy. It asks whether that advantage survives comparison with the true marginal-entropy policy under the identical permutation posterior. It is a new split on each existing public graph, not four new independent datasets or evidence about actual corruption incidents.

## Frozen construction

- Datasets: cached Cora, ogbn-arxiv-20k, OpenAlex-20k, and Amazon Computers. Features and graph edges only; no semantic labels enter scores.
- Seeds: 119003 and 119004, previously unused. For each graph and seed, shuffle all vertices and split 55% anchor, 15% calibration, remainder test. These splits reuse the underlying graphs of earlier experiments and can overlap earlier split membership.
- Use the frozen `study.py` predictor and calibration implementation, copied as `study_frozen.py`. Anchor-neighbor centered features and the same centroid/ridge candidates and temperature grid. Calibrate on disjoint random b=6 calibration blocks exactly as in `study.calibrate`; no b=12 test outcomes choose the predictor or temperature.
- Randomly permute the test fold into disjoint b=12 blocks; independently permute feature slots inside each. Retain zero-anchor rows and all other difficult cases. Use the first min(floor(test_n/12), 220) blocks per graph/seed. One truthful original-slot query per block, fixed across policies. No query costs or network delays are simulated.

## Policies and outcome

All policies use the identical score matrix and selected temperature. Compare (1) exact permutation-posterior marginal entropy, (2) converged Sinkhorn entropy, (3) whole-block expected-success value, (4) uniformly random row, and (5) smallest forced completion gap / highest forced runner-up. Also record no-query and attainable best single-row outcomes.

Exact Gibbs marginals use log-space subset dynamic programming for the permanent, with forward and backward subset tables and all edge probabilities. Sinkhorn uses log-space alternating row/column normalization, maximum 10,000 iterations, and row/column residual target 1e-10; record iterations and residuals. This improves convergence control relative to the earlier 200-iteration exploratory approximation and distinguishes iterative error from the permanent-vs-Sinkhorn approximation.

Whole-block values are ranked through logsumexp of forced-completion weights after removing one MAP-edge constant in every row, preventing cancellation for near-certain blocks. Forced gaps use the minimum non-MAP forced-completion loss. Policy ties use relative tolerance 1e-10 and absolute tolerance 1e-12 and are averaged uniformly over tied rows.

The primary outcome is complete recovery of the original 12-row serialization. Every policy uses the same constrained MAP decoder. If the truthful permutation is a constrained optimum, success is 1 / number_of_tied_constrained_optima; otherwise it is zero. A max-plus/counting subset DP computes optimum scores and the number of optimal completions for every forced edge. Numerical score ties use absolute tolerance 1e-12. The no-query decoder follows the same rule. This explicitly averages decoder ties as well as action ties, rather than using a lexicographic winner. Report exact-decoder tie frequency and counts.

## Validation, reporting, and limitations

Before running graph tests, compare permanent log normalizer, every marginal, forced optimal scores/counts, and expected decoder outcomes against independently enumerated permutations for b=2..8, including signed entries, integer ties, all equal scores, heterogeneous margins, and two temperatures. Abort on discrepancies. On every b=12 test block, additionally compare max-plus forced scores with the independent all-pairs exchange loss formula and check marginal row/column normalization.

Report all graph/seed results without pooling seeds as independent graphs. Primary paired difference: whole-block value minus exact marginal entropy. Also report value minus Sinkhorn, exact entropy minus Sinkhorn, value minus random, and value minus forced-gap. Use 10,000 paired block-bootstrap resamples per graph/seed, percentile 95% intervals, deterministic bootstrap seeds. These are conditional on graph/split and the fitted predictor; they do not capture uncertainty across independently collected graphs and are descriptive, not multiplicity-adjusted simultaneous intervals. Save block IDs, slot order, scores, true and approximate marginals, policy action distributions, decoder counts/outcomes, calibration metadata, source/data hashes, and execution start/end time.

The experiment confirms or challenges a 12-row mechanism under injected known-block permutation faults and a fitted score model. It does not establish an operational incident model, broad novelty, or realistic corruption prevalence. The original eight-task receipt stays frozen; this experiment has a separate receipt and all files remain in `experiments/b12_confirmation/`.
