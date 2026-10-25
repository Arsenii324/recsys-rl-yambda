> Архивный снимок состояния на 25 октября 2026 года; актуальный основной отчёт — [README.md](README.md).

# Current research state — 2026-10-25

## Higher target and hypotheses

The assignment asks whether RL can shift sequential recommendations toward longer-term user benefit while retaining relevance, and how reward design and policy behavior shape that tradeoff. Our offline Yambda logs contain observed listens, not impressions or non-listens; the completion RM predicts a response conditional on a listen. Predicted reward, ranking, and real user benefit are distinct quantities. No causal or retention claim is supported.

Working hypotheses: (1) CE may improve next-item relevance more reliably than optimizing the current RM; (2) lower-temperature reward tilt buys proxy reward at a relevance cost; (3) recurrent history rereading may help by changing the history representation. The recurrent branch currently tests history encoding/optimization, not multi-item slate planning or diffusion. Existing C evidence shows little utility headroom: exact-oracle minus myopic proxy J is about 0.000624. This is not a general negative result for planning or GFN methods.

## Completed evidence

- **Strong reference and history gate (exploratory reused R9 test):** full-supported GPT P0 NDCG@100 .121343; matched CE mean .126152; CE plus newly trained history gate .137340; archived full43 gate .134537. The new gate adds about .01119 over CE and .00280 over the archived gate. Same 100k catalog, 232,049 events, 4,812 UID clusters; 34,520 unsupported targets remain misses. The R9 cohort was already viewed; UID intervals are conditional on two fitted seeds, not fresh confirmation. Organic/recommended slices differ. [Gate summary](artifacts/history-gate-ce-r9-reused-exploratory-20261025/summary.json).
- **Recurrent history encoding (separate H256 R1/R4-source comparison, exploratory reused R9):** R1-source NDCG@100 .146768 mean across seeds 17/29; shared R4-source .149287. Paired R4−R1 is +.002519 (95% UID-cluster CI [.001975,.003099]); organic +.004199, recommendation-driven +.000876. Same 232,049-event order and 100k catalog. Both used 32.768M examples and 13,722,144 parameters, but R4 training took about 1.83× as long, so this is not a compute-matched architecture result. Full paired metrics and source pins are in the [R1/R4 summary](infra/a100/huginn-context256-20261025/output/context256c-compiled-study/r1-r4-source-r9-paired-summary.json).
- **RL reward/relevance sweep:** relative to P0, beta .03/.1/.3 gained +1.62/+0.46/+0.16 percentage points in expected frozen-RM score on a separate 1,024-context cohort. R9 NDCG deltas were −.006485/−.000645/−.000087. The beta .3 UID interval includes zero; equivalence was not tested. These are exploratory fixed-fit comparisons on the reused R9 test, with bootstrap uncertainty conditional on two seeds. They show a reward/relevance tradeoff, not observed user response or a causal effect. [summary](artifacts/rl-beta-pareto-exploratory-20261025/summary.json), [table](artifacts/rl-beta-pareto-exploratory-20261025/summary.csv), and [figure](artifacts/rl-beta-pareto-exploratory-20261025/reward-vs-ranking.png).
- **Continuation proxy:** existing C=16 exact-Q evidence indicates near-saturation and sparse recommended-target coverage (2.2%). The current recurrent study instead tests history representation. See [research map](docs/RESEARCH_MAP.md) and [continuation results](research/results/continuation-q-runset-validation.md).

## Running and next

H256 R1 and shared R4-source now have fixed-final R9 exports on the same cohort; these are recurrent history-encoding comparisons, not slate-planning or diffusion results. H64 four-mode controls, including the untied arm, remain pending: no terminal result is in the local record at this snapshot. The ledger records the H256 A100 and H64 Kaggle scopes and pins in [`docs/experiment-ledger.csv`](docs/experiment-ledger.csv). The H256 comparison is not compute matched; keep the task baseline and RM/RL tradeoff central.

## Limits and provenance

R9 and validation data have been reused repeatedly; no untouched test cohort remains. Test-set bootstrap intervals quantify event-weighted contrasts with UID clustering, conditional on fixed models. The two RL fits do not provide training-seed uncertainty. Catalog OOV misses are retained. RM uses history item embeddings pooled without order and train-derived duration/popularity metadata; it is not a sequence-aware counterfactual response model. See the original [assignment](task-original.txt), [research map](docs/RESEARCH_MAP.md), [RL/P0 comparison](research/results/rl-p0-objective-comparison-20261025.md), and [experiment ledger](docs/experiment-ledger.csv).
