# Воспроизведение и входные данные

Эта инструкция отделяет воспроизводимые шаги подготовки от недостающих внешних весов. Репозиторий содержит код, конфигурации, lockfile, агрегаты и компактные ранги; исходный parquet и основные checkpoints не включены. GPU-прогоны выполнялись на Kaggle T4 и A100 с отдельными окружениями, поэтому CPU lockfile не является установочным рецептом CUDA-среды.

## Данные и каталоги

Использовался официальный набор yandex/yambda, revision dd6f3a19eef5866e346c3270e098baa641a44948, файл flat/50m/listens.parquet. Ожидаемый размер 369 110 332 bytes, SHA-256 eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1.

~~~bash
mkdir -p data/raw/yambda-50m
curl -L --fail -o data/raw/yambda-50m/listens.parquet \
  https://huggingface.co/datasets/yandex/yambda/resolve/dd6f3a19eef5866e346c3270e098baa641a44948/flat/50m/listens.parquet
printf '%s  %s\n' eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1 data/raw/yambda-50m/listens.parquet | sha256sum -c -
~~~

Для full-supported parent используется полный train pool: 41 142 628 целей, train-sample-rate 1.0. Для RL/continued CE использовалась отдельная 10%-ная подвыборка из 4 110 754 целей, seed 17. Они имеют разные train_indices и manifest и не взаимозаменяемы.

~~~bash
python scripts/prepare_sequences.py data/raw/yambda-50m/listens.parquet \
  --output data/processed/yambda100k-fullsupported-eventuniform \
  --catalog-size 100000 --train-sample-rate 1.0 --seed 17

python scripts/prepare_sequences.py data/raw/yambda-50m/listens.parquet \
  --output data/processed/yambda100k-eventuniform \
  --catalog-size 100000 --train-sample-rate 0.10 --seed 17
~~~

Индексы и манифесты зарегистрированных прогонов: full-supported train_indices SHA-256 948a6d33e6a5055c868d65c59b2b81ce60ff856111c68d7659f57008f595c474, manifest aa5570c4e16cbd61e33457bd31970b356c8a49b9938de4ba90c773109bb85570; sampled train_indices SHA-256 90a38ccfc2098bd78299a92c62b6cfffdceab5b354283fe9f095d3bf81fbae3c, manifest ecc0aa56ac64c3d2a0351f4e02c60977035855a41b83ed8d88ad947c00d949b6. После подготовки сравните фактические хеши с этими pins до запуска.

## GPU-окружение

Обычный uv sync --locked устанавливает CPU-сборку PyTorch, так как pyproject/uv.lock фиксируют CPU index. Не запускайте его в рабочем GPU-окружении: он может заменить CUDA-сборку. Проверенные среды были:
- T4: Python 3.13.15, torch 2.11.0+cu130, CUDA 13.0.
- A100: Python 3.12.3, torch 2.11.0+cu128, CUDA 12.8.

Если PyTorch с нужной CUDA-сборкой уже установлен в системном Python GPU-хоста, проверенный способ добавить остальные зависимости, не заменяя torch:

~~~bash
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python3 -m venv --system-site-packages .venv-gpu
UV_PROJECT_ENVIRONMENT="$PWD/.venv-gpu" uv sync --locked --no-install-package torch --python "$PWD/.venv-gpu/bin/python"
.venv-gpu/bin/python -c 'import torch; assert torch.cuda.is_available()'
~~~

Это не устанавливает CUDA PyTorch: он должен быть предварительно настроен хостом. Для существующего GPU venv запускайте его python напрямую или используйте uv run --no-sync --python /absolute/path/to/venv/bin/python. Точный образ GPU-хоста в lockfile не зафиксирован.

## Parent, RM и RL assets

Full-supported GPT parent SHA-256 36b035e517d2792e1e0027b1d27f12f92c29688727d322135159f681ee0b1ea6. Combined RM, использованная в RL-серии, SHA-256 cd2d36b22e03e6fc7e912bba8a09d2a18a9c54cae7c2fae4b928329c9f07f197; history RM SHA-256 34a5053b43db71b42e9fafdcf3bef87fa884aa9d3da8257a332e63f063042d7c. Они выдавались как private Kaggle input dataset arsen4ikvar/recsys-core-aux-20261019; нет в GitHub-публикации.

Скрипт scripts/run_rl_grouped.py требует каталог, заданный через --aux: manifest.json в корне и три файла в корне — checkpoint_gpt_seed43_fullsupported.pt, checkpoint_combined_seed17_primary.pt, checkpoint_history_seed17.pt. В приватном Kaggle input arsen4ikvar/recsys-core-aux-20261019 файлы модели лежат в checkpoints/, а manifest.json — в корне. Отдельного преобразователя нет; локальную flat-структуру нужно собрать копированием:

~~~bash
mkdir -p /absolute/path/to/aux-flat
cp /path/to/kaggle-input/manifest.json /absolute/path/to/aux-flat/
cp /path/to/kaggle-input/checkpoints/gpt_seed43_fullsupported.pt /absolute/path/to/aux-flat/checkpoint_gpt_seed43_fullsupported.pt
cp /path/to/kaggle-input/checkpoints/combined_seed17_primary.pt /absolute/path/to/aux-flat/checkpoint_combined_seed17_primary.pt
cp /path/to/kaggle-input/checkpoints/history_seed17.pt /absolute/path/to/aux-flat/checkpoint_history_seed17.pt
sha256sum /absolute/path/to/aux-flat/manifest.json /absolute/path/to/aux-flat/*.pt
~~~

Ожидаемые checkpoint SHA-256: full-supported GPT 36b035e517d2792e1e0027b1d27f12f92c29688727d322135159f681ee0b1ea6; combined RM cd2d36b22e03e6fc7e912bba8a09d2a18a9c54cae7c2fae4b928329c9f07f197; history RM 34a5053b43db71b42e9fafdcf3bef87fa884aa9d3da8257a332e63f063042d7c. Manifest SHA-256 должен быть 5253a228935b0205f9689ab832da0835e4c2ca7849e73fc562f5b6e1111bb059; feature manifest и checkpoint hashes также закреплены в configs/rl-full43-ce-rloo8-20261025.json. Команда запуска ниже проверяет их до обучения. Приватный доступ к Kaggle input нужен для точного воспроизведения; веса не публикуются.

## Примеры запуска

Команда full-supported baseline:

~~~bash
python scripts/run_experiments.py base \
  --data data/processed/yambda100k-fullsupported-eventuniform \
  --config configs/base-fullsupported-seed43.json \
  --output runs/full43 --device cuda
~~~

Пример повторного запуска RL arms с указанными выше входами:

~~~bash
for seed in 17 29; do
  python scripts/run_rl_grouped.py \
    --data data/processed/yambda100k-eventuniform \
    --aux /absolute/path/to/verified-aux-root \
    --config configs/rl-full43-ce-rloo8-20261025.json \
    --output runs/rl --seed "$seed" --device cuda
done
~~~

Для обучения нужны checkpoints соответствующих parent/RM, подготовленные теми же конфигурациями и хешированными входами. Публичная публикация сохраняет метрики и per-query ранги, но не выдаёт веса и приватные Kaggle assets. Манифест файлов и дополнительные lineage-признаки находятся рядом с результатами под infra/ и research/.
