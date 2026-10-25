# Grouped RL post-hoc validation

This is a post-hoc comparison of frozen step-32,000 checkpoints on the reused 10,000-event seed-991 validation sample and full 100,000-item catalog. The ordered event-index SHA-256 is `950a0d20892fec4b383091752af797bee9d4096a49be97577edc25ed0c7f0a12`; all 10,000 events remain in ranking denominators and 8,624 have in-catalog targets. The SFT reference is checkpoint SHA `4076631ea76eca428fb3747f4f112b33633d7c835048cf7e1b2a2f807533674d`. These intervals are conditional on fitted models and a previously reused validation cohort; they are not independent confirmation, test results, causal response estimates, or off-policy evaluation.

## Full-catalog validation matrix

Each cell reports NDCG@100 / Recall@100. Supported CE is averaged over the 8,624 supported target events; unsupported targets remain misses for ranking.

| Model | All | Organic | Recommended | Supported CE |
|---|---:|---:|---:|---:|
| Frozen SFT reference | .06089 / .18960 | .09369 / .25748 | .02829 / .12218 | 8.5001 |
| CE, seed 17 | .06919 / .20870 | .10560 / .28276 | .03302 / .13514 | 8.3527 |
| CE, seed 29 | .06892 / .20980 | .10532 / .28497 | .03277 / .13514 | 8.3555 |
| Context REINFORCE, seed 17 | .05901 / .18730 | .08969 / .25426 | .02854 / .12079 | 8.5264 |
| Context REINFORCE, seed 29 | .05909 / .18900 | .08926 / .25467 | .02912 / .12378 | 8.5299 |
| RLOO-8, seed 17 | .05974 / .18890 | .09073 / .25607 | .02895 / .12218 | 8.5160 |
| RLOO-8, seed 29 | .05961 / .18830 | .09058 / .25607 | .02886 / .12099 | 8.5153 |

Across the two initializations, CE reached mean all-event NDCG@100 `.06905` and Recall@100 `.20925`, above the frozen SFT reference (`.06089`, `.18960`). Context REINFORCE averaged `.05905` / `.18815`; RLOO-8 averaged `.05967` / `.18860`. RLOO-8 exceeded context REINFORCE by `.000622` NDCG@100 (paired UID 95% CI `[.000056, .001245]`); the per-seed differences were `+.000722` for seed17 (CI `[−.000058,+.001483]`) and `+.000521` for seed29 (CI `[−.000296,+.001316]`), each individually including zero. The Recall@100 difference was `.00045` (CI `[-.00134, .00216]`). Both RL arms were below their seed-matched CE checkpoints: context minus CE was `−.01000` NDCG@100 (CI `[−.01172, −.00840]`) and `−.02110` Recall@100 (CI `[−.02553, −.01652]`); RLOO-8 minus CE was `−.00938` (CI `[−.01111, −.00781]`) and `−.02065` (CI `[−.02497, −.01595]`). CE minus SFT was `+.00817` NDCG@100 (CI `[+.00669, +.00985]`) and `+.01965` Recall@100 (CI `[+.01501, +.02407]`).

The RLOO-8 versus one-draw context REINFORCE comparison changes both draws per context and baseline; by itself it cannot isolate a baseline effect. A matched K=8 context-baseline control has now completed and addresses the draw-count confound. Its preregistered primary contrast was the paired two-seed mean exact-q objective gap on the same fixed 1,024 query rows (full-query denominator); the 878-supported subset was used only where reconstructing the P0 gap requires supported targets. Secondary endpoints were NDCG@100 and Recall@100 on the same 10,000 validation rows and the already-viewed 232,049-row R9 test order. It used the same two fitted seeds, SFT parent, RM, 32,000×128 update dose, and final checkpoint as RLOO-8, with K=8 and history-only RM baseline replacing the RLOO leave-one-out baseline. Both arms make 32.768M main-RM training calls per seed; the context-baseline arm additionally makes 4.096M cheap history-baseline RM calls per seed, so this isolates the baseline more closely but is not an exact FLOP match. The first packaged launch failed before training because child processes could not import `recsys_tt2`; the fresh packaged child-boundary preflight passed and the recovery completed on two visible T4s. Exact-q and CPU ranking results on both reused cohorts are in the [matched-control note](context-reinforce8-control-20261025.md). No seed or checkpoint selection was used.

## Exploratory reuse of the previously viewed R9 test order

The original six step-32k grouped-RL actors and the two matched K=8 control actors were rescored on the exact same canonical 232,049-event R9 test order. The SFT ranks and other baselines were reused from the existing R9 cache, whose event order and UID/target/support/organic columns matched the source event arrays across all rows. A fresh 512-row CPU SFT rank probe matched the cached SFT ranks exactly. Because this cohort and its aggregate outcomes had already been viewed during R9, this is exploratory rescore evidence, not fresh confirmation or a basis for choosing seeds/checkpoints. Unsupported targets remain misses (rank 101); the full 100k catalog was used.

| Model | All NDCG/Recall@100 | Organic NDCG/Recall@100 | Recommended NDCG/Recall@100 |
|---|---:|---:|---:|
| SFT reference (P0) | .09662 / .22176 | .16660 / .32647 | .02816 / .11932 |
| Matched CE, mean of seeds 17/29 | .10424 / .24052 | .17915 / .36264 | .03221 / .14184 |
| Context REINFORCE, mean of seeds 17/29 | .09575 / .21953 | .16546 / .33249 | .02905 / .12844 |
| RLOO-8, mean of seeds 17/29 | .09604 / .22014 | .16573 / .32783 | .02867 / .12358 |
| Context REINFORCE-8, mean of seeds 17/29 | .09591 / .21983 | .16516 / .32249 | .02814 / .11939 |
| Recent frequency (cached R9 comparator) | .10216 / .20403 | .19383 / .36724 | .01247 / .04433 |

The paired two-fit RLOO-8 minus context-reinforce difference was +.000287 NDCG@100 (95% UID CI [+.000163,+.000423]) and +.000605 Recall@100 ([+.000163,+.001057]). Context REINFORCE-8 minus context-1 was +.000153 NDCG@100 ([+.000025,+.000295]) and +.000297 Recall@100 ([−.000132,+.000747]); minus RLOO-8 it was −.000134 NDCG@100 ([−.000226,−.000042]) and −.000308 Recall@100 ([−.000641,+.000042]). These are exploratory reused-test contrasts. All policy-gradient arms fall below matched CE and SFT on all-query NDCG/Recall. Recent frequency exceeds context/RLOO in all-query NDCG but has lower all-query Recall; matched CE exceeds recent frequency on both. The following paired differences use the same per-query ranks and recent-frequency cache, with UIDs resampled in 1,000 paired bootstrap replicates (95% percentile CIs):

| Actor minus recent frequency | Group | Δ NDCG@100 [95% UID CI] | Δ Recall@100 [95% UID CI] |
|---|---|---:|---:|
| Matched CE (two-fit mean) | All | +.002074 [+.000007,+.004281] | +.036494 [+.028941,+.044231] |
| Matched CE (two-fit mean) | Organic | −.014825 [−.019658,−.010450] | −.014587 [−.025761,−.004747] |
| Matched CE (two-fit mean) | Recommended | +.018609 [+.016350,+.020674] | +.086473 [+.076814,+.094805] |
| Context REINFORCE (two-fit mean) | All | −.006408 [−.008729,−.003949] | +.015507 [+.008968,+.022355] |
| Context REINFORCE (two-fit mean) | Organic | −.028810 [−.034843,−.023043] | −.044594 [−.057878,−.032868] |
| Context REINFORCE (two-fit mean) | Recommended | +.015511 [+.013438,+.017401] | +.074311 [+.066042,+.082404] |
| RLOO-8 (two-fit mean) | All | −.006121 [−.008415,−.003678] | +.016113 [+.009467,+.022947] |
| RLOO-8 (two-fit mean) | Organic | −.028369 [−.034307,−.022618] | −.043879 [−.056952,−.031990] |
| RLOO-8 (two-fit mean) | Recommended | +.015647 [+.013546,+.017610] | +.074810 [+.066516,+.082821] |

This indicates the recent-frequency baseline ranks organic events better than the actors, while CE and both policy-gradient actors rank recommended events better; CE also exceeds recent frequency on the overall cohort. These are descriptive ranking contrasts, not counterfactual policy effects. All intervals are conditional on the same two fitted seeds, not seed-population uncertainty.

Other cached R9 baselines are retained as distinct-exposure descriptive context:

| Cached R9 model | All NDCG/Recall@100 | Organic NDCG/Recall@100 | Recommended NDCG/Recall@100 | Exposure note |
|---|---:|---:|---:|---|
| GPT seed43 full-supported | .12134 / .28657 | .21016 / .43296 | .03444 / .14334 | 41.14M-anchor SFT |
| Fixed mixture, γ=.5, full-supported GPT | .12879 / .31178 | .23229 / .50401 | .02752 / .12370 | fixed SFT/history mixture |
| Validation-selected global mixture, γ=.15 | .13114 / .31257 | .23403 / .50483 | .03048 / .12446 | validation-tuned |
| Learned history gate, full-supported GPT | .13454 / .32599 | .23463 / .50345 | .03661 / .15236 | validation-selected |

These cached full-supported and hybrid policies used different training exposure/selection, so they are not matched-dose comparisons to the new actors.

Completion strata use the observed target `played_ratio_pct` for evaluation only (zero: 15,302; nonzero below 50%: 59,051; at least 50%: 157,696). Two-fit mean NDCG@100 / Recall@100 were: SFT `.06555/.19043`, `.05147/.17023`, `.11654/.24410`; CE `.07657/.21736`, `.05972/.19912`, `.12453/.27375`; context `.06437/.19625`, `.05119/.17646`, `.11660/.25237`; RLOO-8 `.06412/.19128`, `.05103/.17177`, `.11660/.24760`, in zero / nonzero-low / high order. These observed-label slices do not estimate counterfactual engagement. Exact all/organic/recommended and completion-stratum aggregates, per-query ranks/labels, lineage pins, and UID-bootstrap contrasts are in [the test export summary](../../infra/attempts/rl-grouped-reused-test-cpu-20261025/rl-grouped-reused-test-summary.json), [paired analysis](../../infra/attempts/rl-grouped-reused-test-cpu-20261025/paired-analysis.json), and [rank NPZ](../../infra/attempts/rl-grouped-reused-test-cpu-20261025/rl-grouped-reused-test-ranks.npz). The export took 1,831.8 CPU seconds; first-512 SFT rank parity was exact. Peak RSS was not recorded; the largest live sample was 1,064 MiB, with no memory or disk issue observed.

The ranking and supported CE measures diverge across the RL arms: mean supported CE was `8.5282` for context REINFORCE and `8.5156` for RLOO-8, versus `8.3541` for matched CE and `8.5001` for SFT. This one-step frozen-proxy objective therefore did not produce a ranking gain over CE on this cohort.

## Exact-q comparison

The saved exact full-catalog q evaluation uses a fixed 1,024-query subset of the same validation cohort (878 supported targets; query SHA `27eebe3388f87343bd8989f1eb0158854e193d59b688de2a84d2aecc64c63e20`). Mean objective gaps were CE `.01746`, context REINFORCE `.003426`, and RLOO-8 `.001722`. The two-seed mean paired gap difference RLOO-8 minus context was `−.001705` (UID-bootstrap 95% CI `[−.001756, −.001654]`). These are objective-fit diagnostics, not observed user utility. Lower q gap for RLOO-8 co-occurred with a small validation ranking increase over context, while both RL arms remained below CE ranking.

## Reconciliation and provenance

The CPU export took `94.288` seconds using two CPU threads. Its SFT per-query ranks match the existing seed-43 cache for the same checkpoint and exact event order: UID/support/organic/event arrays match, maximum per-query NDCG@100 difference is `2.14e-8`, and Recall@100 is identical. CPU and provider GPU event counts/support counts match for every arm. All NDCG@100 and Recall@100 aggregates reconcile within `1e-5`. One top-10 discrepancy exceeds that review threshold for CE seed 17: CPU minus GPU Recall@10 is `−.000100` overall and `−.000201` organic; NDCG@10 is `−.000029` overall and `−.000058` organic. Other top-10 differences are below `1e-5`. The localized drift is retained as measured and is not rounded into a parity claim.

Both seeds completed 32,000 optimizer updates at batch size 128 (4.096 million examples per arm); provider wall time was `4,254.7` seconds with two T4s, or `8,509.4` device-seconds by wall-times-device accounting. Kaggle's quota display changed from 9.17 to 10.37 hours rounded; that display is not treated as exact device-seconds. A prior import-failed attempt is separately estimated at 60 device-seconds. GPU utilization telemetry was not available in the retrieved result.

The per-query NPZ and machine-readable provenance, bootstrap intervals, provider aggregates, and exact-q joins are in [the analysis JSON](rl-grouped-posthoc-analysis-20261025.json). The six frozen checkpoints, worker logs, run results, and q diagnostics are retained under `infra/attempts/reward-frontier-grouped-rl-recovery-v3/gpu-recovery/output/reward-frontier-recovery-v2-20261025/`.

## Exploratory target-completion strata

The saved seed-991 query rows were joined to raw `played_ratio_pct` by exact `event_index` (all 10,000 events are validation rows and UID joins match); the outcome was not supplied to any model. This post-hoc split leaves the all-query primary result unchanged. Low completion `<50%` contains 3,387 events / 1,604 UIDs (including 656 zero-completion events / 421 UIDs); high completion `≥50%` contains 6,613 / 2,597. Unsupported targets remain misses within each stratum. For low / high, two-seed mean NDCG@100 and Recall@100 were: SFT `.05684/.18128`, `.06296/.19386`; CE `.06587/.19914`, `.07068/.21443`; context REINFORCE `.05543/.17936`, `.06091/.19265`; RLOO-8 `.05598/.18010`, `.06156/.19295`. CE exceeded SFT in both strata. Versus SFT, context NDCG differences were −.00142 (95% UID CI [−.00269,−.00008]) low and −.00205 ([−.00309,−.00110]) high; RLOO-8 was −.00086 ([−.00223,+.00054]) low and −.00139 ([−.00245,−.00044]) high. Thus the ranking loss for the policy-gradient actors is not concentrated in low-completion events in this split; zero-event estimates are imprecise. This conditions on observed listens/labels and is neither a causal response estimate nor counterfactual policy value. Full per-policy/per-stratum values, supported by 1,000 UID-cluster paired bootstrap resamples, are in [the strata JSON](rl-target-completion-strata-20261025.json).
