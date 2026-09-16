# CoreFlow independent confirmation: ClassEval method-level analysis

- decision: `COREFLOW_Q185_PRIMARY_NONINFERIOR`
- protocol: `coreflow-indep-confirm-classeval-v1`
- rows per seed: 202
- seeds: 41, 42, 43

## Seed-level correct counts

| seed | Full | CoreFlow q185 | Independent-SVD |
|---:|---:|---:|---:|
| 41 | 51 | 49 | 36 |
| 42 | 47 | 52 | 48 |
| 43 | 41 | 39 | 42 |

## Pooled paired comparisons

| comparison | diff pp | 95% CI pp | noninferiority (-3pp) |
|---|---:|---:|---|
| core_vs_full | 0.17 | [-1.65, 1.98] | True |
| core_vs_isvd | 2.31 | [0.33, 4.29] | True |
| isvd_vs_full | -2.15 | [-3.96, -0.33] | False |

## Sensitivity (pooled CoreFlow - Full)

| margin | CI lower >= margin |
|---|---|
| -1 pp | False |
| -2 pp | True |
| -3 pp | True |
