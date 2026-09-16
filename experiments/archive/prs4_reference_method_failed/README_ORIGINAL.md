# CoreFlow PRS-4A / PRS-4P pilot — AutoDL instructions

## What this package does

This package implements two bounded actions only:

1. **Scheme A** archives the prior Stage-1 `full_vectorized` result of 0/64 as a low-floor failure. It explicitly records that this was not the PRS-v2 Original LoRA-Flow qualification.
2. **Scheme B** evaluates the frozen first 16 already-exposed qualification problems using Original LoRA-Flow, CoreFlow q185, and the frozen padded matched-q185 Independent-SVD runtime with gate seed 41 (48 records total).

The pilot verifies the metric/evaluator pipeline. It cannot qualify LiveCodeBench, cannot open `formal164`, and cannot support method-selection or non-inferiority claims. The formal164 dataset is not included and no runtime path resolves it.

## Required existing AutoDL assets

The host must already contain:

```text
/root/autodl-tmp/Model
/root/autodl-tmp/coreflow_m0
/root/autodl-tmp/coreflow_formal_v1_workspace/m2_assets
/root/autodl-tmp/coreflow_formal_v1_workspace/reports/m2_assets/asset_lock.json
/root/autodl-tmp/coreflow_formal_v2_primary_workspace/isvd_banks/k5_code/matched_q185_bf16
```

The old Stage-1 workspace is optional:

```text
/root/autodl-tmp/coreflow_submission_final_stage1_v1_workspace
```

If it is present, Scheme A verifies and binds the complete raw 0/64 archive. If absent, the package preserves the available console diagnostics and labels the evidence strength accordingly; Scheme B can still run.

## Verify and extract

```bash
cd /root/autodl-tmp
sed -i 's/\r$//' coreflow_prs4a_prs4p_pilot_v1_upload_20260805.tar.gz.sha256
sha256sum -c coreflow_prs4a_prs4p_pilot_v1_upload_20260805.tar.gz.sha256
tar -xzf coreflow_prs4a_prs4p_pilot_v1_upload_20260805.tar.gz
cd /root/autodl-tmp/coreflow_prs4a_prs4p_pilot_v1_upload
```

## One-command full run

Run this in the foreground:

```bash
set -o pipefail
export MODEL=/root/autodl-tmp/Model
export SOURCE_WORK_ROOT=/root/autodl-tmp/coreflow_m0
export M2_ASSET_ROOT=/root/autodl-tmp/coreflow_formal_v1_workspace/m2_assets
export M2_ASSET_LOCK=/root/autodl-tmp/coreflow_formal_v1_workspace/reports/m2_assets/asset_lock.json
export FORMAL_V2_WORK_ROOT=/root/autodl-tmp/coreflow_formal_v2_primary_workspace
export LEGACY_STAGE1_WORK_ROOT=/root/autodl-tmp/coreflow_submission_final_stage1_v1_workspace
export WORK_ROOT=/root/autodl-tmp/coreflow_prs4a_prs4p_pilot_v1_workspace
export CUDA_DEVICE=0

bash scripts/run_prs4a_prs4p.sh all \
  2>&1 | tee "$WORK_ROOT/logs/prs4a_prs4p_all.log"
```

For an unattended run:

```bash
nohup bash -lc '
cd /root/autodl-tmp/coreflow_prs4a_prs4p_pilot_v1_upload
export MODEL=/root/autodl-tmp/Model
export SOURCE_WORK_ROOT=/root/autodl-tmp/coreflow_m0
export M2_ASSET_ROOT=/root/autodl-tmp/coreflow_formal_v1_workspace/m2_assets
export M2_ASSET_LOCK=/root/autodl-tmp/coreflow_formal_v1_workspace/reports/m2_assets/asset_lock.json
export FORMAL_V2_WORK_ROOT=/root/autodl-tmp/coreflow_formal_v2_primary_workspace
export LEGACY_STAGE1_WORK_ROOT=/root/autodl-tmp/coreflow_submission_final_stage1_v1_workspace
export WORK_ROOT=/root/autodl-tmp/coreflow_prs4a_prs4p_pilot_v1_workspace
export CUDA_DEVICE=0
bash scripts/run_prs4a_prs4p.sh all
' > /root/autodl-tmp/prs4a_prs4p_all_console.log 2>&1 &
echo $! > /root/autodl-tmp/prs4a_prs4p_all.pid
```

Typical runtime on one RTX 4090 is about 45–90 minutes; reserve up to 2 hours because LiveCodeBench evaluator time varies by generated program.

## Monitor

```bash
tail -f /root/autodl-tmp/prs4a_prs4p_all_console.log
```

In another terminal:

```bash
cd /root/autodl-tmp/coreflow_prs4a_prs4p_pilot_v1_upload
export WORK_ROOT=/root/autodl-tmp/coreflow_prs4a_prs4p_pilot_v1_workspace
bash scripts/run_prs4a_prs4p.sh status
ps -ef | grep -E 'run_pilot_b|serve_method' | grep -v grep || true
nvidia-smi
```

## Download after completion

```text
/root/autodl-tmp/coreflow_prs4a_prs4p_pilot_v1_results.tar.gz
/root/autodl-tmp/coreflow_prs4a_prs4p_pilot_v1_results.tar.gz.sha256
```

The result archive contains the seal, Scheme-A closure, all 48 raw/evaluator records, the long metric table, the final exploratory summary, and the frozen protocol/data identities.
