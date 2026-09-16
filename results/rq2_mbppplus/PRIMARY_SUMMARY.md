# CoreFlow formal-v2 primary MBPP+ analysis

- decision: `COREFLOW_Q185_PRIMARY_NONINFERIOR`
- protocol: `coreflow-formal-v2-primary-v1`
- rows per seed: 250
- seeds: 41, 42, 43

## Seed-level correct counts

| seed | Full | CoreFlow q185 | Independent-SVD |
|---:|---:|---:|---:|
| 41 | 35 | 36 | 34 |
| 42 | 47 | 48 | 51 |
| 43 | 41 | 38 | 38 |

## Pooled paired comparisons

| comparison | diff pp | 95% CI pp | noninferiority |
|---|---:|---:|---|
| core_vs_full | -0.13 | [-2.40, 2.13] | True |
| core_vs_isvd | -0.13 | [-2.27, 2.00] | True |
| isvd_vs_full | 0.00 | [-1.73, 1.73] | True |
