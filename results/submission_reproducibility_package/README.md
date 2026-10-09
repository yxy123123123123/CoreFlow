# Submission reproducibility package

This directory is the neutral, public evidence entry point associated with the
PeerJ Computer Science submission.  It supplements the experiment-specific
summaries elsewhere in `results/` and contains no benchmark prompts/tests,
model generations, model weights, LoRA/gate checkpoints, or compiled banks.

| File | Public evidence retained |
|---|---|
| `quality_task_outcomes.csv` | Task ID, method, gate checkpoint, binary outcome, evaluator status, and raw-output SHA-256 for MBPP+ and ClassEval. |
| `rtx4080_request_records.json` | The 50 request-level latency and throughput observations for each of nine RTX 4080 SUPER configurations, plus configuration-level memory and power measurements. |
| `qwen_training_lifecycle.json` | Measured group-2/group-3 training times for eight independent LoRAs and CoMoL joint training. |
| `rq1_lifecycle_summary_records.json` | Five compilation totals and three-method direct-load summaries transcribed from the frozen archive. These are summary records, not reconstructed raw logs. |
| `source_audit.json` | SHA-256 and record count for every private raw input used to derive the distributable records. Paths are intentionally omitted. |
| `RELEASE_STATUS.json` | Evidence status, remaining limitations, and version anchor. |

`tools/build_minimal_public_records.py` implements the sanitization step used
for the first three files.  It exports only identifiers, hashes, outcomes, and
numerical measurements.  The private input manifest is not distributed because
it contains machine-specific paths.

The historical directory `results/reviewer_required_v2/` is retained for
protocol provenance.  Its status files now point to this neutral public package.

