# CoreFlow Reproducibility Package

English | [中文](README.md)

This repository organizes the core algorithms, experiment drivers, frozen protocols, and lightweight result summaries used in the CoreFlow paper. It is prepared for release on GitHub and does not redistribute the base model, LoRA weights, gate checkpoints, large compiled banks, raw generations, or third-party benchmark datasets.

## 1. Where Is the Core Code?

| File | Purpose | Category |
|---|---|---|
| `coreflow/decompose.py` | Constructs trace-normalized shared input/output spaces and projects each LoRA update into an expert-specific core matrix | **CoreFlow offline compilation algorithm** |
| `coreflow/runtime.py` | Implements the shared input projection, gate-weighted fusion in the core space, shared output projection, and runtime installation | **CoreFlow online execution algorithm** |
| `coreflow/direct_load.py` | Materializes runtime tensors directly from a standalone deployment artifact without loading the source LoRA bank | Deployment loader |
| `coreflow/io.py` | Loads and validates LoRA factors, configurations, module mappings, and hashes | Asset interface |
| `coreflow/ablation.py` | Implements one-sided sharing, unbalanced decomposition, random bases, and other structural variants | Ablation code |
| `coreflow/isvd.py` | Implements the matched-budget Per-Expert SVD baseline | Baseline code |
| `coreflow/loraflow.py` | Connects the original LoRA-Flow assets and gates to the experiment runtime | Source-system integration |
| `coreflow/statistics.py` | Provides problem-clustered paired bootstrap and related statistical utilities | Statistical analysis |
| `coreflow/evaluator.py` | Evaluates MBPP+ and executable code outputs | Evaluation code |
| `coreflow/classeval_eval.py` | Provides method-level ClassEval evaluation | Evaluation code |

The shortest path through the implementation of the core equations is:

```text
coreflow/decompose.py::build_module_bases
  -> coreflow/decompose.py::project_core
  -> coreflow/runtime.py::coreflow_delta
  -> coreflow/runtime.py::install_coreflow
```

## 2. Mapping from Paper Experiments to Code

| Research question | Experiment | Code | Lightweight results |
|---|---|---|---|
| RQ1 | Full-subspace algebraic correctness, compilation, and direct loading | `experiments/rq1_compilation_correctness/`, `experiments/rq1_direct_load/` | `results/rq1_correctness/` |
| RQ2 | Primary MBPP+ quality evaluation | `experiments/rq2_mbppplus_quality/` | `results/rq2_mbppplus/` |
| RQ2 | Independent ClassEval confirmation | `experiments/rq2_classeval_confirmation/` | `results/rq2_classeval/` |
| RQ2/RQ5 | Formal q224 sensitivity and hidden-state drift diagnostics | `experiments/reviewer_required_v2/` | `results/reviewer_required_v2/`; task-level public records in `results/submission_reproducibility_package/` |
| RQ3 | Throughput, memory, and profiler comparison of LoRA-Flow, CoreFlow, and Per-Expert SVD | `experiments/rq3_system_efficiency/` | `results/rq3_system/` |
| RQ4 | W0--W6 workloads and K=3/5/8 adapter-runtime scaling | `experiments/rq4_workload_k_scaling/` | `results/rq4_scaling/` |
| RQ4 | Qwen3-8B route comparison among Independent-Full, CoreFlow, and CoMoL, plus descriptive RTX 4080 SUPER measurements | `experiments/reviewer_required_v2/` | `results/reviewer_required_v2/derived_statistics.json`; request/lifecycle records in `results/submission_reproducibility_package/` |
| RQ5 | q-grid, A0--A6 ablations, real gates, and controlled routing interventions | `experiments/rq5_ablation_gate/` | `results/rq5_ablation_gate/` |

See `docs/EXPERIMENT_MAP.md` for the script-level mapping, `docs/SOURCE_PROVENANCE.md` for the correspondence between the original experiment packages and this release layout, and `docs/DATA_PROVENANCE.md` for dataset selection, fingerprints, and redistribution boundaries.

The status of early exploratory studies, superseded implementations, and unsuccessful experiments is recorded in `docs/HISTORICAL_EXPERIMENTS.md`. In particular, the PRS-4 code and failure summary are retained under `experiments/archive/` and `results/archive/` and are not mixed with the paper's primary evidence.

## 3. Environment

The frozen environment used for the paper's GPU experiments was:

- Ubuntu Linux and Python 3.12.3;
- NVIDIA RTX 4090 with 24 GB of memory;
- NVIDIA driver 580.105.08 and CUDA 12.4;
- PyTorch 2.5.1+cu124;
- Transformers 4.36.2;
- the PEFT fork distributed with LoRA-Flow, reported as `0.7.2.dev0` at runtime;
- NumPy 1.26.4, Flask 3.0.0, and Safetensors 0.8.0.

Install a PyTorch build compatible with the local CUDA environment first, and then run:

```bash
python -m pip install -r requirements.txt
```

The Transformers and PEFT modifications used by LoRA-Flow are not ordinary PyPI packages. Before running the full generation experiments, prepare the LoRA-Flow source and frozen assets described in `docs/ASSETS.md`.

## 4. Quick Validation

The package can be checked without downloading a model:

```bash
python tools/verify_package.py
python tools/check_syntax.py
```

After modifying the release package, regenerate its integrity manifests:

```bash
python tools/build_experiment_manifests.py
python tools/build_manifest.py
```

If PyTorch is installed, run the synthetic shared-core algebra test:

```bash
python tests/test_shared_core_math.py
```

Full GPU experiments require the following environment variables:

```bash
export PYTHONPATH="$PWD"
export MODEL=/path/to/Llama-2-7b-hf
export SOURCE_WORK_ROOT=/path/to/loraflow_assets
export FORMAL_V1_WORK_ROOT=/path/to/formal_v1_workspace
export FORMAL_V2_WORK_ROOT=/path/to/formal_v2_workspace
export CUDA_DEVICE=0
```

The Qwen/CoMoL route comparison additionally requires `QWEN_MODEL_PATH`, `COMOL_PACKAGE_ROOT`, `QWEN_MATCH_WORK_ROOT`, and the corresponding Python environments. See `experiments/reviewer_required_v2/README_AUTODL.md`. Models, trained weights, and third-party datasets are not redistributed here.

Then enter the corresponding experiment directory and follow its frozen README or driver script. The original scripts retain their historical `/root/autodl-tmp/...` defaults. Prefer overriding those defaults with environment variables instead of silently editing a frozen protocol.

## 5. Levels of Reproduction

This repository distinguishes three levels of reproduction:

1. **Algorithm reproduction:** run the synthetic tensor test to verify the shared spaces, expert-core projection, and online fusion algebra.
2. **Result reanalysis:** read the decision JSON, CSV tables, and protocol seals under `results/` to verify the statistics reported in the paper.
3. **End-to-end rerun:** obtain the third-party model, LoRAs, gates, and datasets; verify their hashes; and rerun the GPU experiments.

The `results/` directory contains audit files, task-level binary outcomes, request-level numerical measurements, lifecycle records, and aggregate summaries rather than model generations. It supports independent reanalysis of the reported statistics but cannot regenerate every token without the external assets.

## 6. Release Scope

This repository releases the core algorithms, principal experiment entry points, protocols, task-level quality outcomes, RTX 4080 SUPER request-level numerical records, Qwen lifecycle measurements, and lightweight aggregate results. The neutral evidence entry point is `results/submission_reproducibility_package/`. It does not include model generations, every raw GPU log, or large intermediate artifacts. The original raw return packages corresponding to the five RQ1 compilation runs and the three-method direct-load summary were not retained; the frozen run-level/aggregate records are published with an explicit descriptive-evidence label.

For the APPS-derived 257-task development set, only task identifiers, source indices, and per-task hashes are released; third-party prompts and tests are not redistributed. The fixed Qwen3-8B revision and configuration/tokenizer hashes are recorded in `docs/manifests/QWEN3_ASSET_MANIFEST.json`, the APPS-derived manifest is in `docs/manifests/APPS_DERIVED_257_MANIFEST.json`, and the full provenance note is in `docs/DATA_PROVENANCE.md`.

The repository is published without an open-source license. See `LICENSE_NOTICE.md` and `THIRD_PARTY_NOTICES.md` before using any material.

See `docs/VALIDATION_REPORT.md` for the checks performed on this organized package and the checks that still require the frozen Linux/PyTorch environment.

## 7. Third-Party Content and Licensing

The base model, LoRAs, gates, and benchmark datasets are not redistributed in this repository. `vendor_canonical/modeling_llama.py` is derived from the Apache-2.0-licensed Transformers implementation and carries the interface required by LoRA-Flow. See `THIRD_PARTY_NOTICES.md` for details.

No open-source license is granted for the CoreFlow-authored code. Copyright remains with the authors; see `LICENSE_NOTICE.md`.
