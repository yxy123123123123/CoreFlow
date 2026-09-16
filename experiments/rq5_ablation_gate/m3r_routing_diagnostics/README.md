# CoreFlow M3R gate diagnostics

This package performs a post-hoc diagnostic analysis of the completed M3-mini K-scaling experiment.

It does not rebuild banks, train gates, or run endpoint generation. It reuses:

- `/root/autodl-tmp/coreflow_m3_k_scaling_mini_upload`
- `/root/autodl-tmp/coreflow_m3_k_scaling_mini_workspace`

Main diagnostics:

- fused-output error across entropy-controlled synthetic gate distributions;
- sparse vs dense gate regimes;
- residual pairwise cosine;
- uniform cancellation ratio;
- whether the M3-mini mixed result is explained by gate-distribution-dependent residual cancellation.

Expected output:

- `coreflow_m3r_gate_diagnostics_results.tar.gz`
- `coreflow_m3r_gate_diagnostics_results.tar.gz.sha256`

