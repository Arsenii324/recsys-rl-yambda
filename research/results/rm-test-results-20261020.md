# Frozen listened-completion test results (2026-10-20)

## Design and provenance

This is a separate, frozen logged-prediction cohort. The ranking selection was
R6 (SHA-256 `62fef8c1d9e89ea55ff1c86b75276f47f306b8ea3debb66862446c810c8cdb8f`);
the RM-only selection was frozen separately in R5 (SHA-256
`1fc599c35a5b346527c2da61153cd6194b0204198201f7b6d64ab6a3db11d3f3`). The
test export used exporter SHA-256
`ceb6a7f1520d1eae8c7ff4f6b3787a7f36bda8125a01057f68b807606a11ef3c`, RM
checkpoint selection remained validation-only, and no test outcome was used to
select or change any model.

The export contains 232,049 chronological test events. Of these, 197,529 have a
target supported by the frozen training catalog and enter conditional
completion-loss denominators; the remaining 34,520 stay in the export without
RM predictions. The supported set has 99,591 organic and 97,938
recommendation-driven events. The label is `min(played_ratio_pct / 100, 1)`
conditional on an observed listen. This measures prediction of logged
completion, not a response to an unshown recommendation.

Each RM was selected by its own hash-pinned run's minimum supported-target
validation MSE. The two full-supported fits have best checkpoints at step
20,000 and final checkpoints at step 80,357 over 41,142,628 training anchors.
Their historical guard remains `running` with unknown return code. The
posthoc closure record establishes that no matching process remained and that
the run/checkpoint/metrics artifacts were complete; it does not establish a
clean guard exit. The exception and all source hashes are in
[`rm-fullsupported-scaling-closure-20261019.json`](rm-fullsupported-scaling-closure-20261019.json).

## Point estimates

Values are event-weighted MSE and MAE on supported test events. Lower is better.

| Predictor | MSE | MAE |
|---|---:|---:|
| Training global mean | .184719 | .405597 |
| Training item mean | .201828 | .403951 |
| History RM, sampled anchors, seed 17 | .168038 | .362536 |
| Item RM, sampled anchors, seed 17 | .189634 | .405934 |
| Combined RM, sampled anchors, seed 17 | .163140 | .350258 |
| Combined RM, sampled anchors, seed 29 audit | .162872 | .350426 |
| History RM, full-supported anchors, seed 17 | .170568 | .358180 |
| Combined RM, full-supported anchors, seed 17 | .167969 | .354169 |
| History-only reader | .167353 | .369241 |
| Pooled-candidate reader | .167625 | .366089 |
| Cross-read R1 | .169544 | .374007 |
| Cross-read R4 shared | .166807 | .369725 |
| Cross-read R4 untied | .166263 | .365207 |

## Paired user-cluster intervals

Intervals below use 1,000 paired user-cluster bootstrap replicates (seed 43).
Each user's events remain together and the metric is event-weighted. Differences
are `comparison − reference`; negative values favor the comparison. These
intervals describe user sampling within this one temporal cohort, not variation
across training seeds.

| Comparison | All supported: ΔMSE [95% CI] | All supported: ΔMAE [95% CI] |
|---|---:|---:|
| Full combined vs sampled combined | +.004829 [−.002715, +.016901] | +.003911 [−.008310, +.023932] |
| Full history vs sampled history | +.002530 [−.001991, +.009464] | −.004356 [−.010221, +.004590] |
| Combined vs history, sampled | −.004898 [−.013558, +.000657] | −.012279 [−.027462, −.003003] |
| Combined vs history, full-supported | −.002599 [−.005999, −.000279] | −.004012 [−.007895, −.001421] |
| Seed-29 vs seed-17 combined, sampled | −.000268 [−.001383, +.000896] | +.000168 [−.002238, +.002116] |
| Untied R4 vs shared R4 reader | −.000544 [−.001724, +.000557] | −.004518 [−.005747, −.003353] |

For full-supported combined versus sampled combined, recommended-event MSE
improved by −.001166 [−.002693, +.000468] and MAE by −.007639
[−.009532, −.005677]; organic-event MSE changed by +.010724
[−.003767, +.032443] and MAE by +.015270 [−.008272, +.051068]. The overall
point estimates therefore do not support a broad test improvement from the
larger anchor exposure; differences vary by event stratum.

For combined versus history within the full-supported cohort, the MSE
differences were −.004911 [−.011211, −.000552] on organic events and −.000249
[−.001228, +.000794] on recommendation-driven events. MAE improved on both:
−.006200 [−.013410, −.001339] organic and −.001786 [−.003008, −.000487]
recommended. Within the sampled cohort, combined-vs-history MAE improved
overall and in both strata; MSE intervals include zero in all three strata.

The direct reader-architecture contrasts (R4 shared versus R1, untied versus
shared, pooled versus history-only, and R1 versus history-only) were computed
after the primary analysis request and were not individually declared in the
pretest comparison plan. Treat them as exploratory paired comparisons. Within
those exploratory comparisons, untied R4 versus shared R4 shows lower MAE
across all, organic, and recommendation-driven strata, while the corresponding
MSE intervals include zero. Shared R4 versus R1 intervals include zero for
both losses in each stratum. Pooled-candidate versus history-only improves
recommendation-driven MSE/MAE, but the overall intervals include zero.

## Interpretation limits

There is bounded evidence that the combined model predicts observed completion
better than history-only within the full-supported cohort. This does not
isolate an item-identity effect: the combined input also contains current-event
duration and train-derived popularity, and the comparison is between fitted
model families. The full-versus-sampled comparisons additionally change
training exposure and selected-step history. The seed-29 audit does not show a
clear difference from seed 17; one additional training seed is not a
distributional replication. These results are descriptive prediction evidence,
not causal action value, preference, engagement, retention, or a basis for
ranking-policy feedback.

## Artifact hashes

- Test per-event export: `022022e6b14d6b5da5aeb0ebc9b3eac75afcf2e17d5752a0d1da7467b16db916`
- Run metadata: `a11675f4d0997374cfe58ad50743c538d8c39cdae9069e94637955d29ceb3385`
- Guard: `f6dadf6b99f56dcea82a490d892c58fb48b606d6eee34af019f1153f55082c19`
- Paired user-bootstrap results, including the exploratory reader contrasts: `a9e0e8ab9c96323abe8b1c87c8f0b4f6541bc6d66cef315bb27b65fcca1fd726`
