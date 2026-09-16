# CoreFlow formal-v2 primary upload

This package runs the CoreFlow formal-v2 primary MBPP+ experiment.

It intentionally does **not** change the formal-v2 primary question:

- Full K5 remains the reference.
- CoreFlow q185 K5 remains the primary method.
- Independent-SVD matched-q185 K5 remains the key fair baseline.
- K=8/16 scaling is excluded from this primary package.

Primary formal matrix:

- `code_full_k5_seed41/42/43`
- `code_core_q185_k5_seed41/42/43`
- `code_isvd_q185_k5_seed41/42/43`

Formal dataset:

- `data/mbppplus_formal_candidate250.jsonl`

Pre-formal gates:

1. runtime and asset preflight;
2. embedded data audit;
3. Independent-SVD bank build and lock;
4. E0 compiler correctness;
5. F0R2 evaluator seal tests;
6. protocol seal before formal model outputs;
7. independent-process ISVD memory smoke audit.

Recommended AutoDL entrypoint:

```bash
bash scripts/run_v2_primary.sh all-primary-2gpu
```

See `AutoDL_v2_formal_primary_执行流程.md` for the full command sequence.
