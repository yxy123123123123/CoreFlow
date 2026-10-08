# Reviewer-required v2 lightweight results

These files are sufficient to audit the aggregate values used by the manuscript while avoiding redistribution of models, checkpoints, benchmark prompts/tests, raw generations, and large runtime artifacts.

| File | Contents |
|---|---|
| `derived_statistics.json` | q224 paired results, Qwen route-comparison summaries, service aggregates, storage/theory values, and the reported drift summary |
| `q224_complete.json` | completion record for the formal q224 sensitivity runs |
| `drift_diagnostics.json` | module-level and condition-level hidden-state diagnostics; no benchmark prompts or generated text |
| `compile_interface_tests.json` | compiler input/output contract checks |
| `storage_audit.json` | logical and stored artifact-size audit |
| `lifecycle_audit.json` | deployment lifecycle audit summary |
| `statistics_required.json` | required statistical checks and derived intervals |
| `PROTOCOL_SEAL.json` | frozen protocol identity and integrity metadata |

The complete GPU return archive is intentionally not stored in Git because it includes large and/or non-redistributable artifacts. The source scripts required to regenerate it are in `experiments/reviewer_required_v2/`.
