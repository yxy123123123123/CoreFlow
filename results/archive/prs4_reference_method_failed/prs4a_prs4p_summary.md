# PRS-4A / PRS-4P pilot result summary

- Pipeline status: `METRIC_PIPELINE_VALID`
- formal164: `CLOSED` and not accessed
- PRS-4Q Original LoRA-Flow qualification: `NOT_RUN`
- Evidence role: exploratory metric/evaluator pilot only

## Three-method first16 results

| Method | Strict | Any test passed | Parse success | Timeout | Max-token truncation |
|---|---:|---:|---:|---:|---:|
| full_legacy_original | 0/16 | 10/16 | 13/16 | 0/16 | 4/16 |
| coreflow_q185_frozen | 0/16 | 10/16 | 13/16 | 0/16 | 8/16 |
| isvd_legacy_padded_matched_q185 | 0/16 | 8/16 | 15/16 | 0/16 | 3/16 |

These 16 already-exposed problems do not qualify the dataset and cannot open formal164.
