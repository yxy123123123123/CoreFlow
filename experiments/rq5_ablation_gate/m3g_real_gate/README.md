# CoreFlow M3G real gate entropy audit

This package runs a post-formal diagnostic audit for CoreFlow:

- no model generation;
- no training;
- no bank rebuild;
- no change to formal-v2 primary results.

It loads the frozen K=5 code gates from formal-v2/M2 assets and measures real gate entropy on the MBPP+ formal prompts. The purpose is to test whether the real serving regime is sparse enough to justify a later K=8 endpoint-quality follow-up after the M3/M3R synthetic diagnostics.

Main output:

```text
/root/autodl-tmp/coreflow_m3g_real_gate_entropy_workspace/reports/m3g_real_gate_entropy_v1/real_gate_entropy.json
```

Packaged result:

```text
/root/autodl-tmp/coreflow_m3g_real_gate_entropy_results.tar.gz
/root/autodl-tmp/coreflow_m3g_real_gate_entropy_results.tar.gz.sha256
```

