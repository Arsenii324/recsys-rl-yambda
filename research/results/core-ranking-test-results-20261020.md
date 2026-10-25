# Frozen core-ranking test results (R9)

## Provenance and estimand

This evaluation used the frozen recommendation selection R6, SHA-256
`62fef8c1d9e89ea55ff1c86b75276f47f306b8ea3debb66862446c810c8cdb8f`.
It contains 14 included policies; no test-driven selection or tuning occurred.
The two ranking shards contain 116,025 and 116,024 events. The merger records
no overlap and a complete ordered union of 232,049 test events. Both shard
exports, merge, and paired bootstrap returned code 0; the guard status is
`completed` and all four stage return codes are 0. The bootstrap used 1,000
user-cluster resamples (seed 41), with event-weighted paired differences.

- Frozen selection: `artifacts/core-controls-cpu-fallback-20261020-r9-20261020T105818Z/selection-frozen.json`
  (SHA-256 `62fef8c1d9e89ea55ff1c86b75276f47f306b8ea3debb66862446c810c8cdb8f`).
- Guard: `artifacts/core-controls-cpu-fallback-20261020-r9-20261020T105818Z.guard.json`
  (SHA-256 `fa443789fb27db709d15938afbe8f70bb56b0b3ce403c8fff9d47caa5a5711b7`).
- Merged per-event export: `frozen-test-rankings.npz` (SHA-256
  `11cc620fcdb7628e26763aba2de9395b9c824df9b25a326d244f3fa69f0eaa6f`);
  metadata SHA-256 `d854f6900c1cac145e450bfe7c7a24e0ee45a1e0fce0f4616b36e50cc60689fb`.
- Paired bootstrap: `frozen-test-bootstrap.json` (SHA-256
  `15497bf44447e1fcbd3adc0fe9c0e660ff2ac582d40b5b6081e5a7f2f8306eb9`);
  metadata SHA-256 `380f6fd083092319333c6c9b93b0c7f2de7a0e79593f0cf8e40835f6c8606162`.

All-target ranking retains every event. The catalog supports 197,529/232,049
targets (85.12%); 34,520 unsupported targets receive rank 101 and remain
misses. Organic has 114,758 events (99,591 supported); recommendation-driven
has 117,291 events (97,938 supported). Thus the strata are not support-matched.

## All-event ranking metrics

| Frozen policy | Hit@100 / Recall@100 | NDCG@100 |
|---|---:|---:|
| Global train popularity | .02871 | .00699 |
| Recent frequency | .20403 | .10216 |
| GPT seed 43, sampled-anchor | .22176 | .09662 |
| GPT seed 17, sampled-anchor | .22074 | .09665 |
| GPT seed 43, full-supported | .28657 | .12134 |
| Fixed history mixture, subset parent, γ=.5 | .28036 | .11859 |
| Tuned global mixture, subset parent, γ=.30 | .28036 | .11958 |
| Learned gate, subset parent | .29109 | .12233 |
| CE, selected validation checkpoint | .22802 | .09909 |
| Constant-baseline RL, β=.01 | .21921 | .09572 |
| Constant-baseline RL, β=.1 | .22126 | .09638 |
| Fixed history mixture, full-supported parent, γ=.5 | .31178 | .12879 |
| Tuned global mixture, full-supported parent, γ=.15 | .31257 | .13114 |
| Learned gate, full-supported parent | .32599 | .13454 |

Recent frequency beats the sampled-anchor GPT models on all-event NDCG@100,
while the full-supported seed-43 GPT beats recent frequency. This latter
comparison also changes training exposure (41.14m versus 4.11m anchors), so it
is not a data-only causal estimate. CE improves over the sampled seed-43 GPT;
the two constant-baseline RL checkpoints are slightly lower overall and on
organic events. Their recommendation-event NDCG differences versus GPT are
small and their registered user-cluster intervals include zero. The test
contains no RM-predicted reward outcome; it therefore does not establish the
reward side of an RL reward/relevance tradeoff.

## Registered paired NDCG@100 contrasts

Differences are policy minus reference. Intervals are 95% user-cluster
bootstrap intervals; each stratum resamples users and weights their events.

| Contrast | All events | Organic | Recommendation-driven |
|---|---:|---:|---:|
| Subset gate − tuned γ=.30 | +.002750 [.002207,.003334] | −.001120 [−.001769,−.000482] | +.006537 [.005771,.007288] |
| Full-supported gate − tuned γ=.15 | +.003393 [.002833,.003918] | +.000594 [−.000010,.001290] | +.006132 [.005446,.006736] |
| Full-supported GPT43 − sampled GPT43 | +.024721 [.021286,.027356] | +.043563 [.036632,.049576] | +.006286 [.005457,.007220] |
| CE − sampled GPT43 | +.002465 [.002108,.002836] | +.003916 [.003190,.004676] | +.001045 [.000807,.001285] |
| RL β=.01 − sampled GPT43 | −.000900 [−.001125,−.000683] | −.001955 [−.002391,−.001535] | +.000133 [−.000093,.000361] |
| RL β=.1 − sampled GPT43 | −.000240 [−.000364,−.000103] | −.000598 [−.000843,−.000372] | +.000110 [−.000035,.000255] |
| Sampled GPT17 − sampled GPT43 | +.000026 [−.000543,.000587] | +.000663 [−.000352,.001726] | −.000598 [−.001070,−.000108] |

Both gates exceed their validation-selected global mixtures overall, but the
subset gate is worse on organic events and better on recommendation events.
The full-supported gate's organic difference is small and its interval includes
zero, while the recommendation-event difference is larger. These frozen-test
contrasts support out-of-sample ranking gains for the gates relative to the
registered scalar controls on this dataset, especially for recommendation
events. They do not establish personalization, causal response, or generality
beyond this single temporal test cohort. They are not a signal for post-test
retuning.

## Relation to validation and scope

Validation selected γ=.30 for the subset parent and γ=.15 for the
full-supported parent. On validation, gate-minus-tuned-mixture NDCG@100 was
+.002409 and +.003531, respectively. The test directions agree, while the
test differences are +.002750 and +.003393. The γ values and checkpoints were
frozen before test scoring; test results did not change them.

The test cohort covers the frozen recommendation registry only. Continuation-Q
and the full-supported RM logged-prediction cohort have separate validation
results and no result in this ranking export. See the separate C and RM records
for their limitations and any further cohort-specific freeze requirements.
