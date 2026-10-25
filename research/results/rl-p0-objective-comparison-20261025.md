# Grouped RL versus the unchanged P0 objective

This is a cached, post-hoc comparison of the six final grouped-RL actors with the unchanged reference policy P0 against the same frozen one-step exact-q objective. It is not a new run. The held-out cohort has 1,024 fixed queries (query SHA-256 `27eebe3388f87343bd8989f1eb0158854e193d59b688de2a84d2aecc64c63e20`); per-query exact-q target probabilities are finite for 878 supported targets. Both actor event-index arrays exactly match the old exact-tilt event-index array. The 878 supported rows span 742 UIDs.

## Exact objective gap

For each supported query, reconstructed the exact normalizer as
`logZ = base_target_logp + rm_target / 0.1 - beta_0.1_target_logq`, then computed P0's gap as `0.1 * logZ - base_rm_expected = 0.1 KL(P0 || q)`. The reverse direction, `KL(q || P0)`, is a different quantity. Reconstructing it from cached `expected_rm / 0.1 - logZ` matches the stored per-query value within maximum absolute error `7.1e-6`.

P0's mean expected reward is `0.660609` over the 878 supported queries (`0.659174` over all 1,024). P0's mean exact gap is `0.0037863` on the matched supported rows (SD `0.001738`; 95% UID-bootstrap CI `[.003653, .003918]`). The 146 unsupported target rows have cached P0 expected reward but no target log-probability, so their per-query exact gaps cannot be recovered. Do not extrapolate the 878-row gap to all 1,024.

Expected RM is available across all 1,024 contexts for both P0 and the actors, but these full-context reward means do not substitute for the matched exact-gap comparison:

| Policy | Mean expected frozen RM, 1,024 contexts |
|---|---:|
| P0 | .659174 |
| CE, two-fit mean | .659288 |
| Context REINFORCE, two-fit mean | .662301 |
| RLOO-8, two-fit mean | .663831 |

| Actor | Seed | Actor gap | Actor − P0 gap (paired 95% UID bootstrap CI) | Mean P0 gap closed |
|---|---:|---:|---:|---:|
| CE | 17 | .016884 | +.013098 [+.012844, +.013371] | −345.9% |
| CE | 29 | .016648 | +.012862 [+.012605, +.013145] | −339.7% |
| CE | two-fit mean | .016766 | +.012980 [+.012729, +.013253] | −342.8% |
| Context REINFORCE | 17 | .003230 | −.000557 [−.000636, −.000476] | 14.7% |
| Context REINFORCE | 29 | .003229 | −.000557 [−.000640, −.000473] | 14.7% |
| Context REINFORCE | two-fit mean | .003229 | −.000557 [−.000636, −.000476] | 14.7% |
| RLOO-8 | 17 | .001642 | −.002144 [−.002235, −.002053] | 56.6% |
| RLOO-8 | 29 | .001637 | −.002150 [−.002240, −.002057] | 56.8% |
| RLOO-8 | two-fit mean | .001639 | −.002147 [−.002238, −.002055] | 56.7% |

Intervals resample the 742 UIDs with replacement, preserving all supported query rows within each sampled UID (20,000 replicates; seed 20261025). They describe query-cohort uncertainty conditional on these fitted actors; they do not estimate training-seed uncertainty. Both policy-gradient arms have smaller exact-q gaps than P0, with RLOO-8 closing more of the gap than single-sample contextual REINFORCE. CE is worse than P0 for this reward-regularized objective. This supports an optimizer-improvement claim versus P0 on the proxy objective, but not an observed-response or long-horizon claim.

## Ranking and behavior context

On the separate reused 10,000-event seed-991 ranking sample, mean NDCG@100 was `.06905` for CE, `.05905` for context REINFORCE, and `.05967` for RLOO-8; the frozen SFT reference was `.06089`. Thus the smaller exact-q gaps did not translate into better next-item ranking than CE, and CE's ranking improvement over SFT does not establish improvement on the proxy objective. This validation cohort was repeatedly consulted and is not fresh confirmation.

The logged 2k/8k/16k/32k validation trace shows CE NDCG@100 rising through 32k; context REINFORCE is flat-to-declining; RLOO-8 is roughly flat from 2k to 32k. This weakens a simple “RLOO only needs much longer training” explanation, but the four validation checkpoints are exploratory.

The separate action audit covers the same 10,000 queries, with 16 common-uniform actions sampled per query. Values below average the two actor seeds. These are generated policy outputs, not actual logged exposures or user responses.

| Model | Entropy (nats) | Top-1 unique items / max share | 16-draw unique items / max share | Top-1 / sampled history-repeat rate | Mean catalog rank: top-1 / sampled | Mean top-1 track duration† |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 8.632 | 2,939 / 3.23% | 49,691 / .186% | 34.19% / 5.64% | 6,637 / 18,462 | not in this action summary |
| CE | 8.702 | 3,801 / 1.25% | 51,320 / .131% | 44.86% / 6.49% | 5,951 / 18,828 | 192.71s |
| Context REINFORCE | 8.548 | 2,747 / 3.50% | 48,331 / .223% | 33.62% / 5.88% | 6,433 / 17,608 | 187.00s |
| RLOO-8 | 8.533 | 2,794 / 3.55% | 48,148 / .222% | 33.75% / 5.89% | 6,636 / 17,749 | 185.93s |

RL policies have slightly lower entropy and similar top-1 repeat rates to P0; RLOO's top-1 concentration is close to P0 and its sampled catalog-rank distribution remains broad. CE has more top-1 item diversity and more history repeats. The worker does not report duration/popularity for each sampled action or P0 top-1 duration in the action-summary artifact; actor top-1 duration is available separately as a catalog-metadata diagnostic, not observed listening time. These summaries do not show real exposure, user reactions, or counterfactual utility.

† Actor top-1 durations are from separate validation policy diagnostics. Duration for the 16 sampled actions and P0 top-1 duration were not emitted in the retrieved artifacts.

## Provenance and compute

The exact-tilt source is `infra/attempts/colab-fullpass-20261019/exact-tilt-policy-beta0p1-per-query.npz` (SHA-256 `eda6aa5b9d284503cb83bf8d5ad99c69ed0c2a9f9a79278c279151279cf1888a`) with sidecar JSON (SHA-256 `0b51c6e3b7a95c9d91010bb24577dc952c3b08f86d5eb84e8674b3cf28399ddd`). It pins P0 checkpoint SHA `4076631ea76eca428fb3747f4f112b33633d7c835048cf7e1b2a2f807533674d`, combined RM SHA `cd2d36b22e03e6fc7e912bba8a09d2a18a9c54cae7c2fae4b928329c9f07f197`, beta `.1`, 100k catalog, and the matching query SHA above. Actor gap arrays are `infra/attempts/reward-frontier-grouped-rl-recovery-v3/gpu-recovery/output/reward-frontier-recovery-v2-20261025/seed-{17,29}/exact-q-gaps-per-query.npz`; each event-index vector exactly equals the exact-tilt cache. Each of the six actors saw 32,000 updates × 128 examples; sampled proxy-reward evaluations total 4.096M per PG actor and 32.768M per RLOO-8 actor, so compute/RM-query budgets differ. Both T4s ran one seed each; provider wall time was 4,254.7 seconds (8,509.4 device-seconds by wall-times-two-device accounting). W&B step reset makes per-arm JSONL/stdout the authoritative curves; it does not invalidate the saved terminal NPZs.
