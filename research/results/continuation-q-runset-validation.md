# Fixed validation results for the continuation-Q run set

All seven seed-17 models use their fixed 2,000-step checkpoints and the same
512 validation contexts. The registered partial six-arm folder remains
partial: its original stop record is immutable, and the separate SubTB untied
completion is referenced only through the run-set manifest. No model or
hyperparameter was selected from these validation results.

| Seed-17 arm | Q MAE | Q RMSE | Raw proxy utility | Path KL | Regularized J |
|---|---:|---:|---:|---:|---:|
| qteacher R1 | .003502 | .004258 | .650171 | .558126 | .644590 |
| qteacher shared R4 | .002399 | .002955 | .650420 | .569962 | .644720 |
| SubTB R1 | .003473 | .004318 | .648076 | .354338 | .644533 |
| SubTB shared R4 | .002630 | .003237 | .648901 | .417659 | .644725 |
| SubTB untied R4 | .002386 | .002911 | .649274 | .449469 | .644779 |
| joint R1 | .003128 | .003848 | .648444 | .383543 | .644609 |
| joint shared R4 | .002507 | .003103 | .648979 | .426306 | .644716 |

The exact soft-Q policy has J=.645086 and the myopic control J=.644345; the
reference policy has J=.63975. The SubTB shared R4−R1 difference is
J=+.0001916 (95% paired user-cluster interval [.0001749,.0002097]), about
3.6% of the exact-Q/reference gap (.00534). Untied R4−shared R4 is
J=+.0000546 ([.0000480,.0000614]); its Q MAE is lower by .0002434. These
intervals resample users over the fixed validation contexts; they do not
measure optimizer-seed uncertainty. The objective gaps are small in absolute
terms, and the C=16 candidate pool covers only 9.96% of validation targets,
so the values do not establish catalog-wide relevance gains or a recurrence
benefit.

A separate seed-29 SubTB pair completed the same 2,000 updates per arm and
512-context validation. Point values are J=.6445845 / Q MAE=.0032091 for R1
and J=.6446646 / Q MAE=.0028069 for shared R4, a J difference of +.0000801.
The seed-29 per-context export and paired user-cluster intervals are pending;
do not pool its points with seed 17 or characterize them as an uncertainty
estimate.

This is a finite-pool proxy diagnostic. It is not an observed recommendation
or user-outcome comparison. The all-target pool ceiling remains about 10%,
with pathway coverage 19.15% for organic and 2.17% for recommended targets.

## Provenance

- Seed-17 paired user-cluster bootstrap: `research/results/continuation-q-runset-bootstrap.json`, SHA-256 `55948a8f16826add886184024391063538685cd0ba8026262ce8bce384c8995b` (10,000 replicates; it pins the then-current seed-17-only manifest SHA `9f1f41095fd3071f1a73ef7c36fb717e78fe4d6a22453e497d90003d1123f51b`).
- Current run-set manifest: `artifacts/continuation-q-runset-20261019/run-set-manifest.json`, SHA-256 `0db90c3b2fff3542fd4221a10aca24b7c907452a50286dfc4e50b253c70e14bd`; it preserves the seed-17 records and adds seed-29 run/checkpoint pins, whose per-context exports remain pending.
- Analysis index: `research/results/continuation-q-runset-analysis-index.json`, SHA-256 `95c179a1fa1e78615e09b201b26a8c4cd00d19d1ae125fa7c07111b113a6de0c`.

The historical seed-17 qteacher R1 run has no saved trainer-source SHA,
guard, or registration; its existing full-512 export pins the real run,
checkpoint, contexts, and validation-Q artifacts. This provenance limitation
is recorded in the run-set manifest. The sections above report validation
results only; the separately frozen test confirmation is documented below.

## Frozen 512-context test confirmation (R5)

The C selection was frozen at 2026-10-20 14:59:17Z before the R5 run began
(15:00:46Z). All nine final step-2,000 checkpoints were fixed in advance; no
model or hyperparameter was selected from test outcomes. The CPU guard completed
in 626.1 seconds with return code 0 at 15:11:12Z. The test sample contains 512
label-blind contexts from 435 users; paired user-cluster intervals use 2,000
resamples. This is a finite-pool proxy comparison, not observed user response.

| Fixed policy | Raw proxy U | Path KL | Regularized J | All-edge Q MAE | Root target R@10 |
|---|---:|---:|---:|---:|---:|
| qteacher R1, seed 17 | .667993 | .531522 | .662678 | .003585 | .0820 |
| qteacher shared R4, seed 17 | .668209 | .540720 | .662801 | .002509 | .0840 |
| SubTB R1, seed 17 | .666003 | .339842 | .662604 | .003533 | .0801 |
| SubTB shared R4, seed 17 | .666811 | .402610 | .662785 | .002675 | .0840 |
| joint R1, seed 17 | .666340 | .365951 | .662680 | .003167 | .0840 |
| joint shared R4, seed 17 | .666884 | .409320 | .662791 | .002552 | .0820 |
| SubTB untied R4, seed 17 | .667130 | .429371 | .662836 | .002386 | .0820 |
| SubTB R1, seed 29 | .666052 | .341289 | .662639 | .003241 | .0879 |
| SubTB shared R4, seed 29 | .666623 | .389672 | .662727 | .002851 | .0859 |

The preregistered primary contrast, seed-17 SubTB shared R4 minus R1, is
ΔJ=+.0001812 (95% user-cluster CI [.0001593,.0002030]), with ΔU=+.0008088
[.0007382,.0008808], ΔKL=+.06277 [.05727,.06892], and ΔQ-MAE=−.0008583
[−.0009723,−.0007391]. The independently initialized seed-29 pair has the
same direction but smaller changes: ΔJ=+.0000871 [.0000743,.0000999] and
ΔQ-MAE=−.0003899 [−.0004788,−.0003000]. These intervals resample users on
the fixed contexts; they do not estimate training-seed variation. The
preregistered qteacher and joint R4−R1 controls also have positive, small ΔJ
and lower Q-MAE, so the result does not show a unique SubTB advantage.

Untied R4 has J=.662836 versus .662785 for shared R4, a descriptive difference
of +.0000512; its Q-MAE is .002386 versus .002675. The direct untied-versus-
shared contrast was not a preregistered contrast and has no registered paired
interval here. Do not call it a winner or infer a sharing effect. The exact-Q
control reaches J=.663118; absolute gaps are .000514 for SubTB R1, .000333
for shared R4, and .000282 for untied R4. The myopic and reference J values
are .662494 and .657941. Thus all learned policies remain below exact Q on
this fixed proxy objective, with small absolute differences.

The fixed pool contains the true target for only 65/512 events (12.7%), or
65/440 catalog-supported targets (14.8%). Coverage is 59/243 organic events
(24.3%) and only 6/269 recommendation-driven events (2.2%). Consequently,
these metrics cannot establish catalog-wide relevance; the coverage ceiling is
especially severe for recommendation-driven targets. The reward is a learned
conditional-completion proxy minus the registered diversity penalty, not
causal utility, retention, or a response to an unshown recommendation.

R5 provenance: selection SHA-256
`f7b62b177fe68f45612788ede1f7ce71bc4c988cb9f61ae15124df2bc64acb7a`, protocol
`aa72ba43652f88ec611217c8045165013ff749375c75649d6d499daccac4cff1`, and
pipeline `77704b9a5b8b04a0ead86290bd6a1cc38da1d201c201fa21f8b4385a2ca66c11`.
Guard SHA-256 is `9a0b3935948c028c3a23ebfe4bd3151f68b827f1771cab9d1be3f21867975405`;
per-context export SHA-256 is
`55f84ba89b6f1a58716140bc9d488f6aabc2a7af65a5684a3d2673eda7d3b854`; summary
SHA-256 is `3838c94ebd96b5a30cfaaedaf33f5b8cbf578580ed2bfa0150a71511a0db817c`.
An earlier R4 attempt accessed event arrays and sampled test indices and began
feature extraction, then failed before C scores or metrics; its exposure is
disclosed in the R4 guard/log. R5 froze GPT parameters before test-array access
and completed without modifying the fixed cohort.
