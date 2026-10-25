# Последовательные рекомендации: supervised baseline, reward model и offline RL

**25 октября 2026**

## Вопрос и вывод

Можно ли улучшить next-item-рекомендации, оптимизируя предсказанный отклик? Supervised-модели, history gate и отдельный R1-source H256 дали сильные результаты на ранее просмотренном R9. RL повысил proxy reward, но сопровождался снижением NDCG. Это offline компромисс, не эффект на реальное поведение.

## Данные и постановка

Yambda-50M: 46 467 212 событий, 9 238 пользователей. Хронологический split с purge; история строго предшествует цели, ties делят контекст. Ранжируется top-100k train-каталог; 34 520/232 049 целей R9 — OOV misses.

Reward model предсказывает clipped played fraction для наблюдавшихся прослушиваний. Она использует историю item-ID и признаки цели; duration/popularity берутся из train-only данных. Показов без прослушивания и propensity для непоказанных объектов нет, поэтому это не причинная модель отклика.

## Основные результаты

| Политика / вариант | NDCG@100 на R9 | Примечание |
|---|---:|---|
| Recent-frequency | .10216 | Строго более ранние события истории |
| Полный supervised GPT, P0 | .12134 | Seed 43, full-supported обучение |
| Matched continued CE | .12615 | Среднее двух seed, 32k шагов |
| CE + обученная history gate | .13734 | Среднее двух seed |
| Архивная full43 gate | .13454 | Ранее обученный контроль |

CE+gate повышает NDCG над CE на .01119 (8.9% относительно CE) и над прежней gate на .00280. Это практическая комбинация, не равнобюджетный контраст: gate получила ещё ~4.1M anchors. Organic NDCG меняется .217140→.237844, recommendation-driven .037129→.039007 (N=114 758/117 291); последняя разница +.001878, 95% UID-cluster CI [.001139,.002703], 3 207 пользователей. Тип события не рандомизированный показ. R9 уже просмотрен; интервалы условны на фиксированных моделях и двух seed.

**Recurrent-архитектура, R9.** R1-source NDCG/Recall@100 .146768/.298127; shared R4 .149287/.299032; untied R4-source .15795585/.306235 (NDCG s17/s29 .157829/.158083). На общем порядке 232 049 событий (test-index SHA `40dec33232ce369bb3e0b9f67dcafc1afd298899543de065b216501944297083`, top-100k, OOV misses retained) shared R4−R1 NDCG +.002519, 95% UID-bootstrap CI [.001975,.003099]; organic +.004199 [.003206,.005262], recommendation-driven +.000876 [.000450,.001336]. Untied−shared NDCG +.008669, both seeds gain +.010220/+.007118. All-target Recall R4−R1 +.000905 [.000047,.001772]. UID bootstrap: 1 000 resamples (4 812/3 552/3 207 users), conditional on fixed fits, not seed CI. При раннем scoring R1/R4 untied ещё обучались; checkpoints были frozen, config/dose неизменны. Все: H=256, 32.768M примеров; R1/shared R4 — 13,722,144 параметра, untied — 14,510,880 (+5.7%). R4 занимал ~1.83× время R1; untied training time близко к shared R4. Не compute-matched. Архитектура — адаптация по мотивам Huginn: source reinjection, prelude/core/coda и residual-sum normalization, не точная репликация. Пины worker/backbone и раннего тестового экспорта — в provenance sidecar; raw hashes сохранены. Это exploratory reuse ранее просмотренного R9, не эффективность recurrence или свежая подтверждающая проверка.

**Reward–relevance sweep.** При β=.03/.1/.3 средняя ожидаемая оценка frozen RM выросла относительно P0 на +1.62/+0.46/+0.16 процентного пункта на отдельной фиксированной выборке из 1 024 validation-контекстов. На том же R9-порядке NDCG изменения составили −.006485/−.000645/−.000087. Интервал β=.3 включает ноль; эквивалентность не проверялась. Reward и ranking относятся к разным когортам. Нельзя трактовать рост RM как фактическое поведение пользователя.

Policy-gradient приблизились к exact q замороженной proxy-цели, но уступили matched CE по ranking. Парные exact-q−P0 и exact-q−RLOO NDCG интервалы включают ноль: ranking-прирост exact q не установлен, а optimizer shortfall полностью не исключён. Exact-q gap — диагностический objective; diversity/repeat измерены на генерациях, не фактических показах.

**Многошаговый механизм.** Отдельный C=16 тест строит горизонт-4 PL slate по фиксированной истории и proxy-utility замороженной RM; target coverage recommendation-driven событий 2.2%, exact-Q даёт лишь .000624 J сверх myopic. Это не сравнимо с H256/H64 recurrent history encoding и не тестирует diffusion. Отдельный H64 four-mode control на момент этой записи ещё не завершён; он не входит в H256-сравнение.

## Интерпретация и ограничения

Практический итог — CE/gate и proxy-reward/relevance компромисс. Без impression/non-click данных нельзя делать выводы о satisfaction, retention, причинной персонализации или отклике на непоказанный объект. R9 и CI по UID уже просмотрены; интервалы условны на фиксированных моделях/двух seed. Full-supported и sampled GPT различаются экспозицией. Это не SOTA-заявление и не отрицательный вывод о целых семействах методов.

## Воспроизведение

Архив включает исходное задание, lockfile, выбранный код/конфигурации и компактные кэшированные результаты, но не сырые данные и основные checkpoints. Источник данных: официальный `yandex/yambda`, revision `dd6f3a19eef5866e346c3270e098baa641a44948`, файл `flat/50m/listens.parquet` (369 110 332 bytes, SHA-256 `eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1`). Получить и проверить его:

```bash
mkdir -p data/raw/yambda-50m
a='https://huggingface.co/datasets/yandex/yambda/resolve/dd6f3a19eef5866e346c3270e098baa641a44948/flat/50m/listens.parquet'
curl -L --fail --output data/raw/yambda-50m/listens.parquet "$a"
echo 'eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1  data/raw/yambda-50m/listens.parquet' | sha256sum -c -
```

**Важное разделение наборов.** Full-supported P0 обучается на всех 41 142 628 train anchors; его `train_indices.npy` SHA-256 `948a6d33e6a5055c868d65c59b2b81ce60ff856111c68d7659f57008f595c474` (manifest `aa5570c4e16cbd61e33457bd31970b356c8a49b9938de4ba90c773109bb85570`). Для него используется `--train-sample-rate 1.0`, не 10%-ный RL pool. Sampled/RL arms используют отдельный `yambda100k-eventuniform`, `--train-sample-rate 0.10`, seed 17; индекс SHA-256 `90a38ccfc2098bd78299a92c62b6cfffdceab5b354283fe9f095d3bf81fbae3c` (manifest `ecc0aa56ac64c3d2a0351f4e02c60977035855a41b83ed8d88ad947c00d949b6`).

```bash
python scripts/prepare_sequences.py data/raw/yambda-50m/listens.parquet --output data/processed/yambda100k-fullsupported-eventuniform --catalog-size 100000 --train-sample-rate 1.0 --seed 17
python scripts/prepare_sequences.py data/raw/yambda-50m/listens.parquet --output data/processed/yambda100k-eventuniform --catalog-size 100000 --train-sample-rate 0.10 --seed 17
```

`uv.lock` намеренно фиксирует CPU-сборку PyTorch; обычный `uv sync --locked` заменит CUDA torch на CPU-сборку. Проверены две разные GPU-среды: Colab T4 full-supported SFT — Python 3.13.15, `torch 2.11.0+cu130`, CUDA 13.0; A100 recurrent worker — Python 3.12.3, `torch 2.11.0+cu128`, CUDA 12.8 из уже установленного `.venv`. Это runtime-записи, не единый install recipe; GPU torch не входит в архив. Для совместимого предустановленного runtime создайте venv с доступом к нему и синхронизируйте зависимости, исключив torch:

```bash
python3 -c 'import torch; assert torch.__version__ == "2.11.0+cu130" and torch.cuda.is_available()'
python3 -m venv --system-site-packages .venv-gpu
UV_PROJECT_ENVIRONMENT="$PWD/.venv-gpu" uv sync --locked --no-install-package torch --python "$PWD/.venv-gpu/bin/python"
.venv-gpu/bin/python -c 'import torch; assert torch.__version__ == "2.11.0+cu130" and torch.cuda.is_available()'
```

Если GPU-пакеты и зависимости уже подготовлены во внешнем окружении, запускайте его Python напрямую или используйте `uv run --no-sync --python /absolute/path/to/gpu-venv/bin/python ...`; для зафиксированной A100 среды это был `.venv/bin/python`. Не запускайте `uv sync` внутри такого окружения.

На полной-supported папке запускается full43 конфиг; reward/политики требуют отдельные указанные checkpoints:

```bash
.venv-gpu/bin/python scripts/run_experiments.py base --data data/processed/yambda100k-fullsupported-eventuniform --config configs/base-fullsupported-seed43.json --output runs/full43 --device cuda
.venv-gpu/bin/python scripts/run_experiments.py reward --data data/processed/yambda100k-fullsupported-eventuniform --config configs/reward-fullsupported-scaling.json --output runs/reward-fullsupported --device cuda
for seed in 17 29; do .venv-gpu/bin/python scripts/run_rl_grouped.py --data data/processed/yambda100k-eventuniform --aux <verified-aux-root> --config configs/rl-full43-ce-rloo8-20261025.json --output runs/rl --seed "$seed" --device cuda; done
```

Полный вариант и RL-вариант намеренно используют разные train-пулы: `--train-sample-rate 1.0` даёт 41 142 628 supported anchors для full-supported parent/RM, а `0.10 --seed 17` — отдельный 4 110 754-anchor пул RL и matched CE продолжения. Это разные training exposure; sampled пул не является сокращённым способом воспроизвести full-supported parent.

`run_rl_grouped.py` сверяет manifest/checkpoint SHA и ожидает в `--aux` manifest вместе с плоскими файлами `checkpoint_gpt_seed43_fullsupported.pt`, `checkpoint_combined_seed17_primary.pt` и `checkpoint_history_seed17.pt`. Соответствующие исходные модели находятся в приватном Kaggle dataset `arsen4ikvar/recsys-core-aux-20261019` (`checkpoints/gpt_seed43_fullsupported.pt`, `checkpoints/combined_seed17_primary.pt`, `checkpoints/history_seed17.pt`), SHA-256 соответственно: `36b035e517d2792e1e0027b1d27f12f92c29688727d322135159f681ee0b1ea6`, `cd2d36b22e03e6fc7e912bba8a09d2a18a9c54cae7c2fae4b928329c9f07f197`, `34a5053b43db71b42e9fafdcf3bef87fa884aa9d3da8257a332e63f063042d7c`. Приватный input нужно получить с Kaggle-аутентификацией и разложить в ожидаемую корневую структуру; одного наличия checkpoint в подкаталоге `checkpoints/` недостаточно. Финальные CE/gate checkpoints не входят в основной пакет: CE seed17 находится по `infra/attempts/rl-full43-ce-rloo8-20261025/gpu/output/rl-full43-ce-rloo8-t4-20261025/seed-17/ce/step-32000.pt` (SHA-256 `352a67f210af1c8b260e3ceeb491c63879d86f5e37672fd375956ac62809a311`), gate seed17 — `infra/attempts/history-gate-on-ce-20261025/results/seed-17/gate-final.pt` (SHA-256 `3e4304b24222bc3c79b8f8ae19b026c7d0a8660eeaa2c7d928012278c4ac4fcb`). Для gate нужны также данные/код и конфигурация обучения; результаты фиксированных моделей не заменяют их.

Per-query ranking NPZ проверяют R9 агрегаты и UID-bootstrap без повторного scoring; они содержат cohort/index/model hashes. R9 уже использовался: это аудит, не новая подтверждающая оценка. H256/R1/R4 worker snapshots, configs и metadata включены, внешний dataset/checkpoints исключены. Проверенные runtime: A100 Python 3.12.3 / torch 2.11.0+cu128 / CUDA 12.8; T4 Python 3.13.15 / torch 2.11.0+cu130 / CUDA 13.0. История: [`REPORT.md`](REPORT.md), [`CURRENT.md`](CURRENT.md), [`docs/RESEARCH_MAP.md`](docs/RESEARCH_MAP.md).

## Результаты и код в комплекте

- R9 ranking и user-cluster интервалы: `research/results/core-ranking-test-results-20261020.md`.
- RM test и границы conditional-label интерпретации: `research/results/rm-test-results-20261020.md`.
- RL/P0 exact-q анализ: `research/results/rl-p0-objective-comparison-20261025.md`.
- Grouped RL и completion strata: `research/results/rl-grouped-posthoc-20261025.md`, `research/results/rl-target-completion-strata-20261025.json`.
- C finite-pool recurrence/proxy evidence: `research/results/continuation-q-runset-validation.md`.
- Machine-readable beta sweep, figure, cached rank arrays, recommended-channel bootstrap, gate summary, configs and source files перечислены в `submission/manifest.sha256`.
- R1/R4-source H256 per-query ranks, [paired three-arm summary](infra/a100/huginn-context256-20261025/output/context256c-compiled-study/three-arm-r9-exploratory-summary-20261025.json), [summary note](research/results/huginn-context256-three-arm-r9-20261025.md) и [early-export provenance sidecar](infra/a100/huginn-context256-20261025/output/context256c-compiled-study/test-export/results/early-export-provenance.json).
