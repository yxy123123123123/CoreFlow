# Reviewer-required v2 experiment package

This directory contains the frozen scripts and protocol used for the final reviewer-requested experiments. It covers:

- formal `q=224` sensitivity on MBPP+ and ClassEval;
- hidden-state drift diagnostics between `q=185` and `q=224`;
- the Qwen3-8B route comparison among Independent-Full, post-hoc CoreFlow, and joint-training CoMoL;
- descriptive RTX 4080 SUPER service measurements;
- compile-interface, storage, lifecycle, and statistical audits.

## Entry points

| Purpose | Entry point |
|---|---|
| Frozen protocol and environment | `config/protocol.json`, `README_AUTODL.md` |
| Minimal required run | `scripts/run_reviewer_minimal.sh` |
| Full reviewer-required run | `scripts/run_reviewer_required.sh` |
| q224 formal sensitivity | `scripts/run_q224.py` |
| CoreFlow/CoMoL route comparison | `scripts/run_comol_groups.py` |
| Drift diagnostics | `scripts/run_drift.py` |
| Compile-interface tests | `scripts/compile_interface_tests.py` |
| Aggregation and statistics | `scripts/derive_statistics.py`, `scripts/summarize_results.py` |

The scripts retain the original Linux paths as defaults where necessary. Set the environment variables documented in `README_AUTODL.md` to use local assets without editing the frozen protocol.

## Publication boundary

The package contains code, generic prompt templates, configuration, and protocol metadata. It does not contain third-party model weights, LoRA/gate/CoMoL checkpoints, benchmark records, compiled banks, or raw model generations. Public lightweight outputs are provided under `results/reviewer_required_v2/`.

The Qwen model identity and hashes are recorded in `docs/manifests/QWEN3_ASSET_MANIFEST.json`. Dataset provenance and the public APPS task-ID/hash manifest are documented in `docs/DATA_PROVENANCE.md`.
