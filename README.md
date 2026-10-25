# RL-дообучение последовательной рекомендательной модели на Yambda

## Краткий вывод

На последовательностях ID треков обучены модель следующего трека и модель вознаграждения (reward model, RM), предсказывающая долю прослушивания. Одношаговое дообучение рекомендателя методом RLOO повысило предсказанную RM-награду, но уступило продолжению cross-entropy (CE) по ранжированию следующей зарегистрированной цели. Этот вывод сохранился при более сильной исходной модели и одинаковом объёме дообучения. В отдельной архитектурной серии четыре независимых прохода по истории дали лучший результат, чем один проход или четыре общих блока.

Все основные тестовые метрики получены на ранее просмотренном тестовом наборе, поэтому это исследовательские результаты, а не свежая подтверждающая оценка и не оценка причинного эффекта на пользователя.

## Данные и постановка

Использован официальный Yambda-50M: 46 467 212 событий, 9 238 пользователей. События сортировались по пользователю, времени и ID трека. Все цели с одинаковой временной отметкой используют одну и ту же предшествующую историю: цель не раскрывает соседние события с равной отметкой. История исходной модели ограничена 64 событиями. Каталог 100 000 треков составлен по частоте только на обучении; равные частоты разрешены по ID. Неизвестные треки истории сохраняются как специальный токен, а цель вне каталога считается промахом и остаётся в знаменателе.

Временные границы в файле Yambda: обучение `timestamp < 25 823 600`; затем 30-минутный purge; валидация `[25 825 400, 25 911 800)`; ещё 30-минутный purge; тест `[25 913 600, 26 000 000]`. Полная валидация содержит 217 014 событий, полный тест — 232 049. Здесь **R9** — этот полный хронологический тестовый порядок; 34 520 целей отсутствуют в train-каталоге. Органические и помеченные рекомендованными события различаются в данных, но это не рандомизированные показы, а журнал непереходов отсутствует. Основная метрика — NDCG@100 по полному каталогу.

Для sampled-пула брались поддержанные train-события с вероятностью .10, независимо по пользователю с seed 17: 4 110 754 примера. Полный пул содержит 41 142 628 целей. Это разные train-индексы и manifest; sampled-пул использовался для одинаковой экспозиции продолжения CE и RL, но не заменяет полный пул исходной модели.

## Вопрос 1. Объяснялся ли ранний RL результат слабой исходной моделью?

Ранний RL уступал обычному обучению на более слабой исходной модели. Чтобы проверить роль качества старта и недостаточного обучения, использовали более сильную P0 и сравнили дообучение CE и RL при одинаковом числе примеров. P0 — GPT-2-подобный каузальный Transformer, обученный с нуля на полном пуле: 100k классов, ширина 128, два pre-LN блока, четыре головы, контекст 64, FFN 4×, dropout .1, связанная входная/выходная таблица item-эмбеддингов и отдельное смещение классов. Обучение: AdamW, batch 256, lr .0002, weight decay .01, 160 714 шагов; на CUDA применялись FP16 autocast и GradScaler. По validation NDCG@100 выбран checkpoint seed 43 на финальном шаге.

| Политика | NDCG@100, R9 | Средняя ожидаемая RM-награда, 1 024 контекста |
|---|---:|---:|
| Частотный ориентир | .1022 | — |
| Исходная модель P0 | .1213 | .6595 |
| Продолжение CE | .1262 | .6597 |
| RLOO-8, β=.1 | .1207 | .6641 |

Продолжение CE подняло NDCG на .0048. RLOO-8 поднял среднюю RM-награду на .0046, но дал NDCG на .0006 ниже P0 и на .0055 ниже CE. Дообучаемые варианты использовали один sampled-пул из 4.11 млн примеров, два seed (17, 29) и 32 000 обновлений по 128 примеров. Для обоих использовались AdamW с lr 2e−5, weight decay 0, ограничение нормы градиента 1 и одинаковая последовательность примеров. Обучались все параметры копии рекомендателя; исходная политика оставалась замороженной. Для RL dropout был выключен, действия сэмплировались из полного 100k softmax при температуре 1. RLOO-8 (REINFORCE с leave-one-out baseline) сэмплировал восемь действий и использовал преимущество `8/7 × (rᵢ − mean(r₁…r₈))` без нормализации награды. В функцию потерь входил точный полный `β KL(πθ || P0)` при β=.1. CE оптимизировала следующий класс. Сравнивались последние шаги, без выбора по тестовым метрикам.

При этой постановке улучшение обычного обучения не переносится на RL: более сильный старт и matched-доза сохраняют преимущество CE. Следующая серия измеряет компромисс между оптимизацией RM и качеством ранжирования.

## Вопрос 2. Оптимизатор ли не достиг цели или цель слабо связана с ранжированием?

RM предсказывает `min(played_ratio_pct/100, 1)` для событий с записанным прослушиванием. Она использует собственные 32-мерные эмбеддинги треков: среднее эмбеддингов истории без порядка, эмбеддинг целевого трека, log-длительность и train-only популярность. MLP имеет размеры 66→128→64→1 с GELU и sigmoid. RM обучалась один проход по sampled-пулу: 8 029 шагов, batch 512, AdamW lr .001, weight decay .0001. Варианты только с историей, только с треком и combined сравнивались по MSE на поддержанных validation-событиях.

Для той combined RM, на которой дообучался RL, post-hoc сверка на тесте дала MSE .1631 против .1680 у history-only RM; MAE составила .3503 и .3625. Ошибка рассчитана на 197 529 поддержанных наблюдавшихся прослушиваниях (4 730 пользователей); для 34 520 OOV-целей RM не выдаёт оценку. Combined-вариант добавляет к истории признаки трека, длительности и train-only популярности. Это улучшает прогноз logged-listen метки относительно history-only модели, при этом вклад этих признаков в персонализированный выбор оценивается отдельно.

Одношаговая оптимальная политика для фиксированной RM равна `q(a|h) ∝ P0(a|h) exp(r(h,a)/β)` и максимизирует `Eπ[r] − β KL(π || P0)`. β задаёт компромисс между предсказанной наградой и отклонением от исходной политики. При β=.03/.1/.3 средняя RM-награда на фиксированных 1 024 validation-контекстах выросла на 1.62/0.46/0.16 процентного пункта; NDCG на тесте R9 изменилась на −.00649/−.00065/−.00009. Интервал для β=.3 включает ноль.

На тех же 1 024 контекстах exact-q получила NDCG .088545 против .089662 у P0; OOV-цели оставались промахами, а 878 поддержанных целей позволяли рассчитать RM-gap. Интервалы q−P0 и q−RLOO включают ноль, а q уступила CE. Это сравнение достигнутой политики с точным оптимумом одного шага именно данной RM. На этой выборке точная оптимизация не улучшила next-item-ранжирование; данные не разделяют полностью вклад ошибки оптимизатора и выбора reward-функции.

В ранней sampled-parent серии сравнивались REINFORCE с одним действием и RLOO-8; затем добавили K=8 с обучаемой базовой оценкой награды `b(h)`, не зависящей от выбранного действия. Это разделяет влияние числа выборок и leave-one-out baseline. На 1 024 контекстах exact-q gap был .003426 для K=1, .002266 для K=8 с `b(h)` и .001722 для RLOO-8. Ранжирование оставалось ниже matched CE. K=8 дал меньший gap в этой серии, но требует в восемь раз больше RM-оценок на контекст. Это отдельный sampled-parent результат, не прямое сравнение с P0 из основной таблицы.

Аудит сэмплировал 16 действий для каждой из 10 000 validation-историй. При β=.03 доля самого частого объекта среди top-1 рекомендаций выросла примерно с .6% у P0 до 2.7%; при β=.3 осталась около .6%. Повтор ранее встречавшегося в истории трека среди сэмплированных действий изменился с 13.4% до 13.8% у β=.1 RLOO-8. Это свойства сгенерированных действий; они не описывают реальные показы или длительность/популярность каждого сэмплированного объекта.

В дополнительной проверке для каждого контекста вход RM заменяли историей другого пользователя, подобранной по длине и квартилю активности; исходную историю P0 оставляли неизменной. При одинаковом среднем KL сравнивали полную и заменённую RM-награду. Их распределения действий различались, но при всех четырёх уровнях KL интервал разницы NDCG включал ноль. На этой выборке изменение RM-истории изменило политику, но не дало установленного различия в ранжировании.

## Вопрос 3. Помогает ли повторное чтение истории и окупаются ли общие веса?

Это отдельная архитектурная ветвь для next-item-ранжирования. Все три режима с самого начала входили в план сравнения: один проход R1, четыре прохода R4 с общими блоками, четыре прохода R4 с независимыми блоками. На каждый вариант приходилось 32.768 млн примеров и два seed. Адаптация вдохновлена Huginn: исходные состояния повторно подаются в последовательные prelude/core/coda блоки.

| H=256 вариант | NDCG@100 | Recall@100 | Параметры |
|---|---:|---:|---:|
| R1: один проход | .1468 | .2981 | 13.72 млн |
| R4: общие блоки | .1493 | .2990 | 13.72 млн |
| R4: независимые блоки | **.1580** | **.3062** | 14.51 млн |

Среднее по двум seed на R9. Shared R4−R1: +.002519 NDCG; untied−shared: +.008669. Независимый вариант имеет на 5.75% больше параметров и похожее время обучения, чем shared R4; R4 занял примерно в 1.83 раза больше времени, чем R1. В этих условиях дополнительные проходы улучшили ранжирование, а независимые блоки дали лучший из трёх результатов. Объём примеров совпадал, вычислительная стоимость — нет.

## Матрица завершённых и незавершённых семейств

Матрица сгруппирована по научным вопросам; полный журнал содержит 49 **логических записей**, включая отдельные fits, пилоты, ошибки и post-hoc экспорты. Все R9-строки ниже — reused test, если не помечены иначе.

| Семейство и варианты | Доза / статус | Наблюдение и вывод |
|---|---|---|
| Временная разметка, каталог и baseline | Полный профиль 46.47M событий; sampled GPT seeds 17/43; полный GPT seed 43. Полный seed17 не завершён подтверждённым финальным checkpoint. | Каталог train-only; частотный R9 .1022, full GPT .1213. Sampled baseline менял каналы по-разному; разница с full не изолирует только объём данных. Локальный KION файл оказался неполным и не использовался как evidence. |
| RM ablation | History, item-only, combined; seed17; один проход/4.11M sampled train listens, validation selection. | Item-only почти совпал с train item mean; combined слегка улучшил validation MSE над history. Это не доказывает, что item effects или user-specific signal отсутствуют. |
| RM logged-listen test | 13 frozen RM вариантов на 232,049 строках; только 197,529 поддержанных listens имеют label/prediction. | Для точного RL RM combined .1631/.3503, history .1680/.3625 MSE/MAE. Полный-supported RM scaling имеет отдельный checkpoint; его числа не подменяют actual RL RM. |
| Ранний RL от слабого sampled P0 | CE, REINFORCE-1, RLOO-8; 2 seed, 32k×128. Затем K=8 с обучаемой базовой оценкой `b(h)`; R9 повторно использован. | Sampled P0/CE/context-1/RLOO-8 NDCG .0966/.1042/.0958/.0960; K=8 с `b(h)` .0959. Этот ранний parent не смешивать с full P0 .1213. K=8 меняет число RM-оценок на действие и baseline относительно K=1. |
| Проверка parent и training dose | Full-supported P0 seed43; continued CE и RLOO-8, seeds17/29, одинаковые 32k×128 sampled anchors. | CE .1262; RLOO .1207 на R9. Мэтч по дозе дообучения и data pool внутри пары, но pretrained P0 имел больший полный train exposure. |
| Reward/KL и exact-q | RLOO β=.03/.1/.3, seeds17/29, 32k×128 на fit; exact q на fixed 1,024 contexts. | Reward рос сильнее при меньшем β, ranking point estimates снижались. q сравнивает optimizer с одношаговым RM-оптимумом, не с долгосрочной целью; q−P0/q−RLOO интервалы включают ноль. |
| RM history interaction и behavior | Историю перемешивали при фиксированном P0 и matched KL; плюс 10k histories ×16 generated actions. | RM history меняла policy distribution, но full-vs-shuffle ranking CIs включали ноль. Наблюдаемое усиление concentration при β=.03 — generated policy property, не real exposure. |
| Исторический gate и γ-mixture | Один обучаемый gate поверх sampled/full P0; global γ-grid; дополнительный gate поверх CE, 2 seed. | Full-P0 gate .1345; CE+gate .1373. Последний получил отдельный дополнительный fit ~4.1M примеров: практичный составной результат, не равнобюджетный контраст. |
| Конечный slate C=16,H=4 | Exact Q/DAG и SubTB варианты на фиксированных малых пулах, validation-only; R5 test guard не выдал scores. | Целевое покрытие рекомендованных событий около 2.2%, exact Q дал лишь +.000624 J над myopic. Это конечный slate-planning proxy, не свободный пятишаговый autoregressive rollout. |
| Matched support и frozen readers | R5 subset-vs-full от одного parent: 32k×256, validation-only. From-scratch support battery: 2 seed, 128k×256. Reader v7/robustness: 2 seed, 8,029 шагов/arm, validation-only. | R5 full−subset NDCG +.000447, CI включает ноль; from-scratch +.006572 на validation, но recipe другой. Reader shared-R4−R1 среднее +.000084, CI включает ноль. Ни один результат не является R9 подтверждением. |
| Повторное чтение истории H=256 | R1/shared R4/untied R4, seeds17/29, 32.768M примеров на fit. | R9 means .1468/.1493/.1580; более ёмкий untied вариант лучший, sharing advantage не показан. R4 по времени ~1.83× R1. |
| H=64 и reader-инфраструктура | H64 R1source/R4source, 2 seed, 32.768M на fit; readers v3/v4 failed pre-model, v6 stopped at ETA gate, v7 validation-only. | H64 fits завершились, но post-fit scorer ожидал 32,768 вместо зарегистрированных 32,000 шагов; test-index values не читались для scoring, test metrics нет. Техническая ошибка не считается отрицательным результатом. |

## Ограничения и следующий вопрос

Тест R9 использовался при разработке; пользовательский bootstrap отражает неопределённость по фиксированным моделям, а два seed не характеризуют полный разброс обучения. Данные содержат зарегистрированные события, но не полный журнал показов и отказов, поэтому измеряется прогноз поведения, а не причинный эффект рекомендации. В RL проверялась только указанная clipped-listen награда и одношаговый выбор действия. Свободный пятишаговый rollout, альтернативные reward-функции и онлайн-эксперимент не проводились. Архитектурная ветвь H256 проверяет повторное кодирование истории для ранжирования следующего трека.

Долгосрочную цель не обучали. Следующий прогнозный тест мог бы оценивать возврат в течение 24 часов: сравнить action-conditioned `r(h,i)` с history-only `r(h)`, строить labels отдельно в каждом split и включать только события с полным окном до границы (`t+24h ≤ cutoff`). Это всё равно прогноз наблюдаемого поведения; причинная оценка потребовала бы онлайн A/B-теста.

## Воспроизведение

Публичный пакет не содержит исходный parquet и крупные веса. Команды ниже воспроизводят подготовку, baseline, RM и обучение CE/RLOO с нуля. Это не бит-в-бит реконструкция сохранённых checkpoints: новые веса меняют checkpoint hashes и итоговые метрики. Публичная копия содержит R9 per-query rank массивы и агрегаты, но не универсальный R9 scorer для новых checkpoints; ниже показаны команды для проверки сохранённых результатов.

### Среда и данные

Проверенный Kaggle T4: Python 3.13.15 / PyTorch 2.11.0+cu130 / CUDA 13.0. A100: Python 3.12.3 / PyTorch 2.11.0+cu128 / CUDA 12.8. `uv.lock` фиксирует CPU PyTorch. На GPU-хосте, где CUDA torch уже установлен в системном Python:

```bash
python3 -m venv --system-site-packages .venv-gpu
UV_PROJECT_ENVIRONMENT="$PWD/.venv-gpu" uv sync --locked --no-install-package torch --python "$PWD/.venv-gpu/bin/python"
.venv-gpu/bin/python -c 'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available())'
PY="$PWD/.venv-gpu/bin/python"
mkdir -p data/raw/yambda-50m
curl -L --fail -o data/raw/yambda-50m/listens.parquet \
  https://huggingface.co/datasets/yandex/yambda/resolve/dd6f3a19eef5866e346c3270e098baa641a44948/flat/50m/listens.parquet
printf '%s  %s\n' eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1 data/raw/yambda-50m/listens.parquet | sha256sum -c -
$PY scripts/prepare_sequences.py data/raw/yambda-50m/listens.parquet --output data/processed/yambda100k-fullsupported-eventuniform --catalog-size 100000 --train-sample-rate 1.0 --seed 17
$PY scripts/prepare_sequences.py data/raw/yambda-50m/listens.parquet --output data/processed/yambda100k-eventuniform --catalog-size 100000 --train-sample-rate 0.10 --seed 17
sha256sum data/processed/yambda100k-eventuniform/manifest.json
sha256sum data/processed/yambda100k-fullsupported-eventuniform/train_indices.npy
```

Ожидаемые значения: sampled manifest `ecc0aa56ac64c3d2a0351f4e02c60977035855a41b83ed8d88ad947c00d949b6`; full `train_indices.npy` `948a6d33e6a5055c868d65c59b2b81ce60ff856111c68d7659f57008f595c474`. Прямой запуск `prepare_sequences.py` создаёт массивы и train-index для полного пула, но его manifest не совпадает с зарегистрированным `aa5570c4…`: в опубликованном run manifest есть дополнительные поля происхождения full-supported overlay. Для нового локального запуска используйте SHA сгенерированного manifest; приведённый ниже фрагмент записывает его в RL-конфигурацию. Не подменяйте им hash исторического run.

### P0, reward model и RL

```bash
$PY scripts/run_experiments.py base --data data/processed/yambda100k-fullsupported-eventuniform --config configs/base-fullsupported-seed43.json --output runs/full43 --device cuda
$PY scripts/run_experiments.py reward --data data/processed/yambda100k-eventuniform --config configs/rm-cpu-full-epoch.json --output runs/rm --device cuda
mkdir -p runs/aux
cp runs/full43/base-best.pt runs/aux/checkpoint_gpt_seed43_fullsupported.pt
cp runs/rm/rm-combined-best.pt runs/aux/checkpoint_combined_seed17_primary.pt
cp runs/rm/rm-history-best.pt runs/aux/checkpoint_history_seed17.pt
```

Следующий фрагмент создаёт manifest, соответствующий новым весам, и локальную копию RL-конфига с их реальными SHA. `run_rl_grouped.py` проверяет feature manifest, aux manifest и SHA каждого checkpoint; manifest здесь является JSON с перечнем весов, его SHA записывается в конфиг.

```bash
$PY - <<'PY'
import hashlib, json
from pathlib import Path
root=Path('runs/aux')
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
names=('checkpoint_gpt_seed43_fullsupported.pt','checkpoint_combined_seed17_primary.pt','checkpoint_history_seed17.pt')
files={n:sha(root/n) for n in names}
(root/'manifest.json').write_text(json.dumps({'schema':'local-retraining-v1','files':files},sort_keys=True,indent=2)+'\n')
cfg=json.loads(Path('configs/rl-full43-ce-rloo8-20261025.json').read_text())
cfg.update(base_checkpoint_filename=names[0],base_checkpoint_sha256=files[names[0]],combined_rm_checkpoint_sha256=files[names[1]],history_rm_checkpoint_sha256=files[names[2]],aux_manifest_sha256=sha(root/'manifest.json'),base_training_manifest_sha256=sha(Path('data/processed/yambda100k-fullsupported-eventuniform/manifest.json')))
Path('runs/rl-config-local.json').write_text(json.dumps(cfg,indent=2)+'\n')
PY
for seed in 17 29; do
  $PY scripts/run_rl_grouped.py --data data/processed/yambda100k-eventuniform --aux runs/aux --config runs/rl-config-local.json --output runs/rl --seed "$seed" --device cuda
done
```

Each trainer writes its validation progress, selected/final checkpoints, and run metadata into the stated output directory. For the historical exact RL actors, the private Kaggle auxiliary dataset was `arsen4ikvar/recsys-core-aux-20261019`; its required files are named in the commands above and have SHA prefixes P0 `36b035e5`, combined RM `cd2d36b2`, history RM `34a5053b`. The exact historical assets are not in this public repository. The from-scratch recipe avoids a private-weight blocker but produces new hashes/results.

### Проверка сохранённых результатов

```bash
sha256sum -c manifest.sha256
python -m json.tool infra/attempts/rl-full43-ce-rloo8-20261025/gpu/output/rl-full43-ce-rloo8-t4-20261025/operator-result.json >/dev/null
$PY - <<'PY'
import numpy as np
p='artifacts/rl-full43-final-actors-test-rank-export-20261025/per-query-test-ranks.npz'
with np.load(p,allow_pickle=False) as z:
    for key in ('rank_parent_full43','rank_ce_seed17','rank_rloo8_seed17','rank_ce_seed29','rank_rloo8_seed29'):
        r=z[key].astype(np.float64); hit=z['supported'] & (r<=100)
        print(key, float(np.where(hit,1/np.log2(r+1),0).mean()))
PY
```

Эта проверка пересчитывает NDCG@100 из включённых rank-массивов (OOV как ноль) для опубликованных checkpoints. Сам test-scoring runner и эти веса не включены; повторный scoring новых checkpoints требует отдельного совместимого scorer. Подробные протоколы, статусы незавершённых работ и машинные provenance pins приведены в [хронологии экспериментов](docs/EXPERIMENTS_RU.md) и [49-строчном логическом реестре](docs/experiment-ledger.csv), но для понимания выводов README не требует открывать эти файлы.

## Файлы проекта

- [Исходное задание](task-original.txt); [машиночитаемая RM-сверка](research/results/rm-actual-combined-checkpoint-reconciliation-posthoc-20261025.json).
- [Архивный подробный журнал](REPORT.md) и [снимок состояния](CURRENT.md); [контрольные суммы пакета](manifest.sha256).
