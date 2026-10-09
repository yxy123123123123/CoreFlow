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

Task-level outcomes, RTX 4080 SUPER request records, and Qwen group-2/group-3 lifecycle timings are published under `../submission_reproducibility_package/`. The complete GPU return archive is not stored in Git because it includes large and/or non-redistributable artifacts. The source scripts required to regenerate the experiment are in `experiments/reviewer_required_v2/`.

`derived_statistics.json` is retained as the frozen aggregate snapshot generated with the original experiment package. For current completion and artifact-availability status, use `statistics_required.json`, `lifecycle_audit.json`, and `storage_audit.json`; these files supersede status fields embedded in the frozen snapshot without altering its reported numerical results.
