# CoreFlow Submission-Final Stage-1 v1

This package implements only the preregistered pre-formal stages:

1. E0-R: freeze and audit LiveCodeBench v4-v6 easy/medium splits;
2. Full/seed41 qualification on 64 frozen tasks;
3. E1: build and validate an independent ragged ISVD-compact bank;
4. E1D: audit Full, Core and ISVD direct-load paths in independent processes.

It intentionally contains no command capable of generating the frozen formal164 split. A later E2A package may be built only after `stage1_decision.json` permits it.

Frozen decisions:

- protocol: `coreflow-submission-final-stage1-v1`;
- K=5, q=185, seeds 41/42/43;
- qualification floor: Full must solve at least 6/64;
- formal split: 164 tasks, unopened in this package;
- paper positioning: Memory-Efficient Dynamic Multi-LoRA Deployment;
- no automatic retry and retained partial outputs stop the stage.

See `docs/AutoDL_Stage1_执行手册.md` for commands.
