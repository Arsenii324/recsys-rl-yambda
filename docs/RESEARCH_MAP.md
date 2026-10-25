# Research map

**As of 2026-10-25.** This is the current scientific understanding layer, not a
run chronology. See [the approved research position](RESEARCH_POSITION.md) for
ownership and decision process; run-level evidence remains in the linked
result notes, hypotheses, and experiment ledger.

## Task target and authority

The task is to build and evaluate a next-item recommender, a reward/response
model, and policy optimization, then discuss ranking, behavior, multi-step use,
and practical limits with reproducible evidence. The source task is
[`task-original.txt`](../task-original.txt); interpreted data/proxy choices are
in [`task-interpretation.md`](task-interpretation.md). The assignment allows
negative RL results and does not equate generated recommendations with user
responses. Optional AI-generated research notes are proposals, not task
requirements or evidence.

## Mechanism model and frontiers

**Working chain:** observed logged anchors and catalog support → history and
ranking representations → predicted response/utility → policy optimization or
planning → practical relevance and cost.

**Anchor-validity frontier:** how much of the event population can a candidate
set represent, and what does a conditional listen label say about unseen
recommendations? Logged completion predicts outcomes conditional on observed
listens; it does not identify response to an unshown action.

**Mechanism frontier:** can reward-guided planning improve decisions beyond
ranking or simple one-step controls? C evaluates approximation to a fixed,
learned utility on a constructed C=16 slate. Its recurrence gain is a model
approximation result; practical planning value is bounded by the gap between
exact policy and myopic control and by target coverage.

These frontiers connect through support and proxy validity, but need not run as
one serial pipeline. A mechanism comparison is useful when the anchor frontier
shows a plausible transfer path; bounded diagnostics may proceed in parallel
when they can change prioritization.

## What the evidence says

| Claim | Evidence type and current reading | Boundary / link |
|---|---|---|
| Full-supported GPT can rank above recent-frequency. | **Observed:** seed43 full-supported NDCG@100 .12134 vs .10216 on R9; sampled GPT is below frequency on NDCG. | Training exposure differs tenfold; seed17 full-supported replication is a resource omission, not a negative result. [`ranking report`](../research/results/core-ranking-test-results-20261020.md) |
| Context helps predict logged completion. | **Observed:** full-supported combined RM improves over its history-only model: ΔMSE −.002599 [−.005999,−.000279], ΔMAE −.004012 [−.007895,−.001421] (comparison minus reference). | Combined features include item, current duration, and popularity; full-fit historical guard exit is unverified. This is predictive, not causal. [`RM report`](../research/results/rm-test-results-20261020.md) |
| Combined RM has same-history action-discrimination signal. | **Exploratory observed:** on 10,000 fixed same-bin validation pairs, combined delta has lower difference-MSE than item-only (−.004593, 95% UID CI [−.005695,−.003471]) but higher MSE than zero/history delta (+.001329 [.000729,.001956]). Adding combined delta to duration+rank+item in fixed UID-folded Ridge does not help (+.0000327 [−.0000191,.0000830]). | Selected observed-listen pairs do not identify causal action effects. No incremental signal beyond metadata/item in this design; global item effects may remain. This reuses validation after earlier summaries, not independent confirmation. [`tie-pair analysis`](../research/results/rm-tie-pair-decomposition-exploratory-20261020.json) |
| A learned history gate beats a validation-tuned global scalar. | **Observed:** both gates exceed their own-parent tuned global mixture on R9 overall; improvements concentrate on recommendation-driven events. | One temporal cohort; does not establish personalization or causal response. [`ranking report`](../research/results/core-ranking-test-results-20261020.md) |
| Recurrence improves finite-pool proxy approximation slightly. | **Observed:** seed17 SubTB shared R4−R1 ΔJ +.0001812 (95% user-cluster CI [.0001593,.0002030]); seed29 +.0000871 ([.0000743,.0000999]). Both reduce all-edge Q-MAE and raise KL. | Other R4 controls (qteacher, joint, untied) also improve slightly; untied−shared +.0000512 is descriptive and unregistered. No unique SubTB/sharing winner. [`C results`](../research/results/continuation-q-runset-validation.md) |
| Recurrence has practical planning value. | **Unresolved:** exact-Q J .663118 vs myopic .662494 leaves only .000624 mean-J headroom in this utility and pool; shared R4 improves over R1 by .000181. | Model-approximation gain is only part of the planning gap. The 16-item pool covers 65/512 test targets (12.7%): 59/243 organic (24.3%), 6/269 recommended (2.2%). [`C results`](../research/results/continuation-q-runset-validation.md) |
| Wider candidate pools create more planning headroom. | **Observed, exploratory:** myopic-IS estimates of `Vhat−J` decrease from .000754 at C=16 to .000536 at C=64 and .000462 at C=256; estimates are stable over nested M and coverage rises. | Finite-sample ordinary-IS plug-in estimate, 32 validation UIDs, downward bias in Vhat; each width renormalizes reference PL, so width does not isolate action-set effects. This does not justify expensive wider-pool neural training under the current proxy. [`planning-gap note`](../research/results/continuation-q-planning-gap-myopic-is-20261020.md) |
| One-step RL improves reward and relevance together. | **Unresolved / not established:** the low-temperature exact proxy raises predicted reward while lowering validation relevance; R9 ranks frozen policies but does not score their reward. | Frozen predictors may score generated actions, but that is predicted reward, not observed behavior. New held-out data alone cannot resolve counterfactual response without exposure or online evidence. [`REPORT`](../REPORT.md) |
| Grouped RL improves the frozen exact proxy objective and next-item ranking. | **Mixed observed results:** on reused 10k validation, matched CE NDCG@100 `.06905`, context REINFORCE `.05905`, RLOO-8 `.05967`, versus SFT `.06089`. On a distinct exact-q subset (1,024 queries; 878 supported), RLOO reduces P0’s mean `0.1 KL(P0||q)` gap from `.0037863` to `.0016393 (−.0021470; 95% UID CI [−.0022378,−.0020550]); context reduces it to `.0032294 (−.0005569 [−.0006365,−.0004756]). On the previously viewed R9 test order, exploratory NDCG@100 is SFT `.09662`, matched CE `.10424`, context `.09575`, and RLOO-8 `.09604`; this is reused-test evidence, not fresh confirmation. | RL ranking remains below matched CE; ranking and q cohorts differ; intervals condition on two fits. RLOO-8 versus one-sample context changes both draw count and baseline. Matched K=8 history-baseline control completed on two T4s. Mean exact-q gap is .002266 on 1,024 rows (versus one-draw context .003426 and RLOO-8 .001722); on the supported 878-row P0 reconstruction it closes 43.4% of P0 gap. Validation ranking CIs do not separate K=8 from either context-1 or RLOO-8; the reused R9 test contrast is exploratory and still below CE. This is frozen-proxy optimization evidence, not user response. Proxy optimization is not user response or policy-class optimality. [`ranking and q note`](../research/results/rl-grouped-posthoc-20261025.md), [`P0 comparison`](../research/results/rl-p0-objective-comparison-20261025.md) |
| Full-supported seed17 reproduces seed43. | **Unresolved due resource failure:** only an interim checkpoint was retrieved; no eligible final model/gate/grid. | Do not infer either success or failure of replication. [`handoff`](handoff.md), [`ledger`](experiment-ledger.csv) |

R5 C evaluated nine fixed models on 512 contexts from 435 users. Intervals
resample users in those contexts; they do not estimate model-seed variation.
Coverage limits the recommendation-path interpretation especially strongly.
The R4 attempt accessed arrays and sampled indices before failing pre-score;
R5 then completed on the frozen cohort. Both exposures are recorded. These
results may inform hypotheses and prioritization, but cannot tune or select
models or serve again as independent confirmation.

## Active hypotheses and discriminating predictions

Detailed definitions are in [`hypotheses.md`](hypotheses.md); these labels
summarize current predictions, not expected outcomes.

- **H1, ranking/data exposure:** one equal-nominal-exposure continuation pair
  from a shared seed43 parent yielded full-supported minus repeated-subset
  NDCG@100 +.000447 (95% paired UID CI [−.001641,+.002698]) and Recall@100
  +.00460 ([−.002128,+.011105]) on the reused fixed validation sample. This
  does not show a clear overall advantage and does not establish equivalence.
  One sampler seed cannot measure training-seed uncertainty or explain the
  earlier from-scratch exposure contrast. See the
  [R5 result](../research/results/matched-data-continuation-r5-validation.md).
- **H2, conditional response signal:** combined inputs should predict logged
  completion better than history alone. It did on full-supported test, but not
  as a clean item-identity effect; duration and popularity differ too.
- **H3, reward-guided policy:** if proxy reward rises while ranking/coverage
  falls, the objective tradeoff persists; if both improve on a preregistered
  cohort, policy value becomes more plausible. Neither identifies actual
  response without exposure or online evidence.
- **H4, recurrence:** the C=16 test shows a small proxy gain over one pass, but
  other R4 controls are similar and exact-Q/myopic headroom is only .000624.
  The validation width diagnostic estimates no larger headroom at C=64/256
  (.000536/.000462), despite higher target coverage. This weakens the case for
  expensive wider-pool neural continuation under the current utility; it does
  not rule out different objectives or estimators. [`C results`](../research/results/continuation-q-runset-validation.md), [`planning-gap note`](../research/results/continuation-q-planning-gap-myopic-is-20261020.md)
- **H5, actor context baseline:** not run. Test only if actor-to-target gap is
  still the decision bottleneck; score variance alone does not predict gradient
  variance.
- **H6, gate adaptation:** test result favors the gate over the selected global
  scalar on this cohort; subgroup gains and single-cohort evidence limit scope.
- **H7, full-supported replication:** seed17 resource failure leaves the
  replication question open, not negative.

Evidence labels used here: **source** = assignment or code contract;
**theory** = derivation or objective definition; **observed** = completed
validation/test artifact; **proposed** = unrun comparison.

## Current decision and transfer gap

**Current decision:** candidate support is no longer a binary reason
to stop mechanism research. On the fixed seed-991 validation sample, both
candidate width and the full43 scorer increase target coverage. At C=256,
full43 mixed coverage is 40.89% overall and 28.14% recommended, compared with
35.44% and 23.48% for subset43. Pure history lifts organic coverage but adds
almost no recommended coverage over full43 GPT at C=256. These curves support
a plausible wider-support path, while remaining selection-conditional
validation evidence. The sampled-versus-full base contrast changes both
training exposure and anchor diversity, so it does not identify their separate
effects. Full tables and provenance are in
[`validation-candidate-support-curve-20261020.md`](../research/results/validation-candidate-support-curve-20261020.md);
the companion figure is
[`candidate-support-width-validation.png`](../research/figures/candidate-support-width-validation.png).

The matched-update R3 attempt failed during its automatic throughput gate
before any final training arm. Its 200-step pilots produced a severe
single-worker versus dual-worker throughput discrepancy; the registered
projection exceeded the wall and device budgets. No final model, validation
metric, or scientific comparison exists. This is an orchestration failure and
leaves the anchor-diversity hypothesis untested. The retrieved artifact records
and cleanup uncertainty are summarized in
[`matched continuation failure`](../research/results/matched-data-continuation-20261020.md).

The completed width diagnostic found increased target coverage but estimated
similar or smaller proxy planning headroom as C grows. Together with C's small
test gain and near-saturated exact-Q/myopic gap, this makes expensive wider-C
neural continuation low priority under the current utility. It does not
abandon the mechanism frontier. A bounded reward/action-signal diagnostic
remains a candidate if ranking and response-proxy evidence diverge; the
fixed-pair validation analysis found no incremental combined-RM contribution
beyond duration, rank, and item-score Ridge features. The failed R3 run showed
that a long multi-worker pilot can consume the available budget before final
arms begin. Root has therefore selected a narrower first comparison: one
matched sampler-seed pair, with the same seed-43 subset parent and fresh
optimizer, comparing repeated sampling of the 4.11M subset anchors with
sampling from 41.14M full-supported anchors. Both arms receive 32,000 nominal
full batches; the final step is primary and AMP skips are reported without
exposure compensation. This asks whether broader anchors help under equal
nominal optimizer exposure. The shared seed is a minibatch-sampler seed for two
fits from one common pretrained parent, not an independent pretraining
replicate or paired training examples. UID bootstrap intervals pair evaluation
queries conditional on this trained pair; they do not measure training-seed
uncertainty. A positive result supports this broader-anchor continuation
recipe; a null does not explain away the earlier from-scratch full-data result.
The R4 source, resource guards, and cost passed independent review, and root
authorized that exact bundle. Submission/admission and outcomes remain separate
provider-record facts; no retry result is assumed here. The fixed validation
sample has informed prior analyses, so this is a validation comparison, not
fresh confirmation.
See the failure record above for R3 provenance.

This comparison is prioritized over wider-C neural continuation because
increasing C raised target coverage but did not increase estimated planning
headroom under the current proxy: the validation estimate fell from .000754 at
C=16 to .000462 at C=256, with finite-importance-sampling and 32-UID limits.
This defers expensive wider-C training for this utility; it does not reject
recurrence under another objective or estimator.
The next-item reread branch has a reviewed, pinned protocol/config/runner but
is not profiled or allocated. It tests whether reading frozen causal GPT
states improves full-catalog next-item CE beyond a final-state adapter; this
is an alternative future frontier, not an active run. [`protocol proposal`](../research/proposals/next-item-reread-protocol.md)

Candidate generation and postprocessing used pinned validation cache/data and
subset/full43 checkpoint inputs. The initial candidate loop's 9,472 count was
a periodic progress snapshot; each array was marked complete after the full
10k loop and flush, and later array checks confirmed all rows. OOV targets
retain the cached `-1` sentinel rather than the safe class-zero gather value.
The analysis reused validation that already informed model/gamma selection;
it is not independent confirmation.

A separate same-timestamp tie-pair diagnostic used a fixed, label-blind
validation pair set and fixed reward checkpoints. It compares observed
completion differences for tied logged listens, with UID-grouped bootstrap and
fixed-fold Ridge contrasts. It was planned after earlier result summaries and
is explicitly exploratory, not independent confirmation. Combined RM
pair-difference MSE is lower than item-only, but higher than zero/history; it
adds no observed OOF improvement to the duration/rank/item baseline. The
estimand remains associative prediction on observed listens; timestamp ties
can hide within-bin order and latent context. See the fixed plan at
[`rm-tie-pair-decomposition-exploratory-20261020.json`](../research/preregistrations/rm-tie-pair-decomposition-exploratory-20261020.json)
and source/data details at
[`reader-tie-pair-validation.md`](../research/results/reader-tie-pair-validation.md).

Predictions: if broader candidate construction raises target coverage across
channels, design a realistic-support recurrence transfer comparison; if
recommended-event coverage stays low, enlarging the C pool alone is unlikely
to support recommendation-path claims; if coverage is adequate but target
prediction remains poor, prioritize the bounded reward/action-signal diagnostic
before mechanism training. These are decision rules, not findings.

The ambitious frontier remains available, but broader-pool neural continuation
is deferred under the current utility and validation planning-gap evidence.
The matched-data continuation produced a small, uncertain overall validation
difference for the one seed-17 sampler pair: full-supported minus repeated
subset NDCG@100 +.000447 (95% UID CI [−.001641,+.002698]). Treat this as no
clear advantage under this continuation recipe, not equivalence and not an
explanation of the earlier from-scratch result. The fixed validation cohort has already informed prior analyses. The conditional
next-item reader v7 completed both registered 8,029-step passes after its profile
gate passed. Shared R4−R1 all-query NDCG@100 averaged +.000084 across two
sampler seeds (95% paired UID interval [−.000416,+.000555]); this is no clear
additional recurrence benefit for this fixed-encoder recipe, not equivalence or
independent confirmation. No test-specific indices were sampled and no test
metrics were produced. Full feature arrays, including held-out bytes, were
staged and hash-verified. See the
[reader result note](../research/results/next-item-reader-conditional-v7-validation.md).
Any claim must distinguish observed prediction, finite-pool proxy utility, and
actual user response.

## Compute boundary

User constraints are use of local/Kaggle/Colab rather than DataSphere and
protection of other projects/checkpoints. The current eight-GPU-hour project
ceiling is an internal provisional accounting cap, not a user hard limit or
guaranteed provider availability. Current spend/reservation accounting remains in
[`allocations.json`](../infra/colab/allocations.json); do not describe the
accounting remainder as available capacity. Provider quota, admission, actual
use, and throughput are distinct. The TPU pilot was source-reviewed but
queued/deleted without hardware admission, so TPU capability remains untested.

## From-scratch support breadth (paired validation)

The two completed equal-compute runs show a positive full-supported minus repeated-subset NDCG@100 difference for each initialization: +0.005030 (seed17, 95% UID-cluster CI [+0.002161,+0.007705]) and +0.008114 (seed29, [+0.005162,+0.010779]). Averaging event deltas across the two fitted initializations gives +0.006572 (1,000 UID-cluster resamples, 95% CI [+0.004442,+0.008625]); uncertainty is conditional on these two fits, not initialization variance. The paired validation sample was reused and this is not fresh confirmation. It supports broader training support helping under this equal-compute from-scratch recipe; the earlier warm-start continuation had no clear gain, and the differing recipes do not isolate why. Seed29 R4 was stopped at 2,500 steps after a stale stop instruction and is excluded; its fresh R5 replacement used the frozen worker and science recipe. See [result note](../research/results/fromscratch-support-battery-r4-r5-validation.md) and [analysis JSON](../research/results/fromscratch-support-battery-r4-r5-validation-analysis.json).
