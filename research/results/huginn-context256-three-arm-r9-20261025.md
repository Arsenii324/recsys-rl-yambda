# Context-256 recurrent backbone: three-arm R9 supplement

**Completed:** 2026-10-25 20:55:57 UTC, before the 20:59 UTC submission deadline. This is an exploratory reuse of the previously viewed canonical R9 test set, not fresh confirmatory evidence.

| Mode | Mean NDCG@100 | Mean Recall@100 | Mean train seconds / fit | Parameters |
|---|---:|---:|---:|---:|
| R1source | 0.146768 | 0.298127 | 1,939.5 | 13,722,144 |
| R4source (shared core) | 0.149287 | 0.299032 | 3,557.8 | 13,722,144 |
| R4untiedsource | 0.157956 | 0.306235 | 3,635.5 | 14,510,880 |

Each fit used 32,768,000 examples (32,000 steps, batch 1,024, context 256), AdamW with constant 2e-4 learning rate, betas (0.9, 0.999), FP16. Applied/skipped updates were: R1 seed17 31,989/11, seed29 31,988/12; shared R4 seed17 31,987/13, seed29 31,988/12; untied R4 seed17 31,987/13, seed29 31,988/12.

Paired UID-cluster bootstrap, 1,000 resamples, two-fit mean, event-weighted NDCG@100 differences (95% percentile intervals): shared R4−R1 **+0.002519** [0.002019, 0.003120]; untied R4−shared R4 **+0.008669** [0.007455, 0.009807]. Untied R4−shared R4 was +0.014325 [0.011761, 0.016765] on organic events and +0.003135 [0.002432, 0.003994] on recommended events. These intervals are conditional on the trained fits and do not quantify seed uncertainty.

The untied mode has 5.75% more parameters than shared R4 and similar fit time. The results show a positive recurrence contrast at equal example count and a higher reused-test score for the untied mode; they do not establish compute efficiency or a source-reinjection effect. R4self was omitted before test access, so that mechanism remains untested. The initial four R1/shared-R4 evaluations occurred before the untied fits finished; their raw metadata is qualified by `early-export-provenance.json`. The final two untied evaluations followed completion of all six fits. All six exports have identical event, UID, target-support, and organic/recommended arrays.

Compact metrics, paired intervals, training counters, parameter counts, provenance pins, and all six result/rank file hashes: [three-arm summary JSON](../../infra/a100/huginn-context256-20261025/output/context256c-compiled-study/three-arm-r9-exploratory-summary-20261025.json).
