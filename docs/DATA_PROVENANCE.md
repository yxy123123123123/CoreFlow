# Dataset provenance and redistribution boundary

This repository records dataset identity and selection logic without redistributing third-party prompts, tests, or generated model outputs.

## APPS-derived 257-task development set

The development set was derived from the `test` split of `codeparrot/apps` at revision `21e74ddf8de1a21436da12e3e653065c5213e9d1`.

The deterministic selection rule retained the first 257 source-order tasks that:

1. were labelled `introductory`;
2. did not define `fn_name`;
3. provided at least five unique short string-input/string-output tests; and
4. could be normalized to a `solve(input_str) -> str` interface.

For each selected task, two tests were exposed to the generation template and at most 20 additional tests were reserved for evaluation. The public manifest contains only source indices, task identifiers, and hashes.

| Object | SHA-256 |
|---|---|
| source file | `5b003a65ac40feb47dd5eaec267a767a6fc435bdcfa68ff715fe869f948e760c` |
| converted source | `60da4062af19241ec9d8fe2b32243dcdd1ec025c22476a6ac17e3f0f7586ad90` |
| frozen JSONL | `0f1bb2decc94a9f19c947d7b1ebf458380648ea183dbcdc2e72c6303a49fd163` |
| task order | `098772db5a8cd0b864d45f2af588658bb26a22d71ec696d4aa1b344873c77933` |
| prompt collection | `920af4f90052c4b8fffe362b7f6d5e6684a54efee6a4b87bafdb400028a92a49` |
| collection fingerprint | `95790f1fb0b6c98cf3cf2049a5b85c2f6ab9ead190243cd81b5ad1472f9edca5` |

The exact normalized-prompt overlap was 0/257 against both the 250-task MBPP+ formal subset and the 202-task ClassEval formal subset. The audit applied Unicode NFKC normalization, lowercasing, whitespace collapse, and SHA-256 equality. This is an exact-string audit and is not a claim of semantic non-overlap.

The public identifier/hash manifest is `docs/manifests/APPS_DERIVED_257_MANIFEST.json`.

## Confirmatory benchmark fingerprints

| Benchmark artifact | SHA-256 |
|---|---|
| frozen MBPP+ formal subset | `049c4fd2aab859f87806d127794e80cc22cab9eb45f71ba6b654efc1a4d7d594` |
| frozen ClassEval formal subset | `03c8e7dc1170ed3c35cbeed5c68b585bc098e6ad63f0882664688f54712cdf57` |

These hashes identify the authors' frozen local evaluation artifacts. They are provided for auditability, not as redistributed copies of the benchmark data.

## What is intentionally absent

- third-party prompts, tests, and reference solutions;
- full raw generations and execution traces;
- base-model, LoRA, gate, and CoMoL weights;
- compiled deployment banks and other large binary artifacts.

Users must acquire each external asset under its original terms and follow the preparation scripts and protocol files in this repository.
