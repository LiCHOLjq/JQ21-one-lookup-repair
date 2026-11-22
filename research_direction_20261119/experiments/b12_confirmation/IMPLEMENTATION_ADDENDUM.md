# Implementation-only acceleration

The frozen Python reference completed its exhaustive oracle and Cora/OGBN-Arxiv runs, then spent several minutes on OpenAlex's strict 10,000-iteration Sinkhorn cap. The coordinator authorized a compiled kernel or independent shards without altering the algorithm or convergence rules.

`sinkhorn_kernel.cpp`, compiled with `g++ -O3 -std=c++17 -shared -fPIC` and without fast-math, implements exactly the same alternating log row/column normalizations, maximum 10,000 iterations, residual target 1e-10, and final row normalization. The permanent DP, MAP counting, policies, data, seeds, calibration, ties, bootstrap, and all reported evaluations stay in the frozen `confirm.py`.

Before the accelerated copy runs, `accelerated_runner.py` compares compiled probabilities and tied entropy action distributions against the Python reference on six independent signed matrices and every available completed block cache from the initial attempt. It aborts on probability differences above 1e-10 or changed tied action distributions. It records all validation cases and hashes in `kernel_validation.json`. Calibration is reconstructed from the frozen source solely to recover each cached run's temperature; outcomes do not choose the implementation.

The initial reference attempt and its files are retained. The accelerated copy writes a separate `accelerated_results/` directory, with its own execution receipt. This is a runtime adaptation, not a changed study design or additional empirical condition. Any unfinished reference process may be terminated only after the accelerated copy completes and passes checks, with that termination recorded.
