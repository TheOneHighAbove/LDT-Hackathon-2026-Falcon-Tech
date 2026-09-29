# Falcon Tech Vehicle ReID

Финальное offline-решение для поиска одного физического автомобиля в static
gallery. Номерные знаки, `camera_id`, порядок CSV и признаки других query не
используются.

Репозиторий сдачи: <https://github.com/TheOneHighAbove/LDT-Hackathon-2026-Falcon-Tech>

## Быстрый старт из чистого клона

Репозиторий не содержит выданные изображения и CSV. Каталог `dataset/`
содержит только описание формата. Перед запуском положите данные так:

```text
dataset/
├── images/
├── train.csv
├── test_query.csv
└── test_gallery.csv
```

Или оставьте датасет вне репозитория и задайте абсолютные пути. Для Docker
удобнее всего использовать переменные окружения:

```powershell
$env:DATASET_DIR = "D:\data\falcon-tech"
$env:OUTPUTS_DIR = "$PWD\outputs"
docker compose build --pull reid-api
docker compose run --rm --no-deps reid-api
```

```bash
DATASET_DIR=/data/falcon-tech \
OUTPUTS_DIR="$PWD/outputs" \
docker compose run --rm --no-deps reid-api
```

После завершения обязательно запустите verifier:

```powershell
.\.venv\Scripts\python.exe -m src.verify_score_optimized `
  --output-dir "$env:OUTPUTS_DIR\score_optimized_speed"
```

На Windows GUI запускается двойным щелчком по [`START_GUI.cmd`](START_GUI.cmd).
При первом запуске файл сам создаёт `.venv` и устанавливает единственную
GUI-зависимость из `requirements-gui.txt`; для этого первого шага нужен доступ
к PyPI. Модель и Docker inference после сборки полностью offline.

## Релизные профили

| Профиль | Locked mAP@10 | Rank-1 | Rank-5 | Веса | Назначение |
|---|---:|---:|---:|---:|---|
| `quality` | **0.841086** | 0.812875 | 0.936684 | 422.9 MB | максимум качества |
| `speed` | 0.836126 | 0.804762 | 0.926279 | **244.6 MB** | итоговый профиль сдачи |

`speed` является профилем контейнера по умолчанию. Он убирает CLIP и его два
reranker-веса, использует DINOv2 при 224 px и два общих Pillow crop; OSNet,
DINOv2, ConvNeXt и все gallery-only операции сохраняются. Измеренная разница
с quality-профилем составляет около `0.0053 mAP`, поэтому для сдачи выбран
более быстрый `speed`.

Quality-профиль запускается явно:

```powershell
.\.venv\Scripts\python.exe -m scripts.infer_score_optimized `
  --release-config configs/score_optimized.json `
  --output-dir outputs/score_optimized
```

Финальный speed-профиль:

```powershell
.\.venv\Scripts\python.exe -m scripts.infer_score_optimized `
  --release-config configs/score_optimized_speed.json `
  --output-dir outputs/score_optimized_speed
```

Обе команды требуют CUDA. Все модели запускаются один раз на изображение,
независимые backbone исполняются в отдельных CUDA streams, а DINO/CLIP local
tokens берутся из того же forward.

## Архитектура

Speed-профиль состоит из трёх extractor-веток:

1. специализированный OSNet: identity, metric и part descriptors;
2. адаптированный DINOv2 ViT-B/14: CLS, regional и patch tokens;
3. ConvNeXt-Tiny: global и local descriptors.

После первичного cosine retrieval применяются компактные top-25/top-50
reranker-модели, DINO patch matching, family GNN, LambdaRank и связи только
внутри static gallery. Search и reranking не входят в performance-тайминг по
официальному протоколу. Quality-профиль дополнительно включает один проход
CLIP ViT-B/16 и CLIP top-25 reranking; flip-TTA отсутствует.

## Open-set отказ

Отказ принимает отдельная логистическая модель корректности именно того top-1,
который выдал финальный reranker. Восемь признаков включают итоговый score и
его top-1/top-2/top-5 разрывы, cosine-сигналы выбранного кандидата, его
плотность и изолированность в static gallery. В каждом эпизоде берётся ровно
один query на identity: 246 известных и 62 неизвестных, то есть 20.13% unknown
при gallery из 750 изображений. Логистическая модель и порог выбираются только
на пяти tune seed с group-OOF по identity. Порог `0.630795` — лучшая точка по
candidate F1 среди вариантов с `TNR >= 0.94`; пять confirmation seed не
участвуют ни в обучении модели, ни в выборе порога.

| Locked confirmation | Значение |
|---|---:|
| F1 | 0.836127 |
| TNR | 0.951613 |
| `0.7 × F1 + 0.3 × TNR` | **0.870773** |

На тех же 20%-эпизодах предыдущая политика давала `0.862576`; прирост новой
калибровки составляет `+0.008196` официального компонента и не меняет ranking.

Отказ кодируется отсутствием query в `candidates.csv`. Количество принятых и
отклонённых test query записывается в manifest и не трактуется как оценка
скрытого test-качества. Полный протокол, OOF-диагностика и сравнение с прошлым
порогом находятся в `outputs/score_optimized_speed/refusal_calibration.json`.

## Артефакты и проверка

Инференс создаёт:

- `submission.csv` — без заголовка, ровно десять gallery ID для каждого query;
- `candidates.csv` — только принятые top-1 ответы;
- `embeddings.npy` — итоговые global descriptors;
- `gallery_index.npz` — static gallery index;
- `inference_manifest.json` — профиль, метрики, время, размеры и SHA-256 всех
  весов и результатов.

Fail-closed проверка схем, top-1 consistency, чисел, хешей и лимита 2 GB:

```powershell
.\.venv\Scripts\python.exe -m src.verify_score_optimized `
  --output-dir outputs/score_optimized_speed
```

Verifier дополнительно проверяет точный порядок query, уникальность top-10,
соответствие `candidates.csv` финальному top-1 и общий размер весов
244 640 363 байта.

## Производительность

Официальная граница включает JPEG I/O, decode, BBox crop, preprocessing,
forward, postprocessing и L2 normalization; search/reranking исключены. Полный
балл: batch-1 не более 40 ms и лучший throughput не менее 100 FPS.

Актуальный локальный замер выполнен на NVIDIA GeForce RTX 3070 Ti Laptop GPU
строго по протоколу из вопросов-ответов: 50 warmup, 300 синхронизированных
batch-1 запусков с медианой и по ≥10 секунд для batch 1/8/16/32. В границу
включён также пятизонный HSV-дескриптор.

| Batch-1 median | Лучший throughput | Batch | Peak CUDA |
|---:|---:|---:|---:|
| 51.27 ms | **105.48 FPS** | 32 | 555.2 MiB |

Это аппаратно-зависимый локальный результат, а не прогноз баллов на A5000
организаторов. Полный отчёт сохраняет все четыре throughput-замера. Запуск:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_release_score_optimized
```

Результат сохраняется в `outputs/score_optimized_speed/benchmark.json`.
Почему ранний замер показывал 20/20, а строгий — меньше, и что реально можно
ожидать от RTX A5000, разобрано в
[`docs/PERFORMANCE_AUDIT.md`](docs/PERFORMANCE_AUDIT.md).

## Диагностика и open-set

Для финального профиля подготовлены воспроизводимые материалы:

- [`docs/SOLUTION_ARCHITECTURE.md`](docs/SOLUTION_ARCHITECTURE.md) — единое
  описание архитектуры, истории выбора, отклонённых идей, ограничений,
  производительности и roadmap;
- [`docs/FINAL_SUBMISSION_CHECKLIST.md`](docs/FINAL_SUBMISSION_CHECKLIST.md) —
  итоговая сверка обязательных артефактов и команд сдачи;
- [`docs/REFUSAL_POLICY.md`](docs/REFUSAL_POLICY.md) — протокол 20% unknown,
  OOF-калибровка и обоснование порога;
- [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) — количественный и
  визуальный разбор ошибок;
- [`docs/SCALING_PLAN.md`](docs/SCALING_PLAN.md) — переход к галерее порядка
  миллиона изображений без изменения query semantics;
- [`docs/TRAINING_REPRODUCTION.md`](docs/TRAINING_REPRODUCTION.md) — порядок
  подготовки split, cache, backbone, reranker и release-экспорта.

Error analysis пересобирается точным release runtime и не участвует в
инференсе:

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze_release_errors
```

## Прототип и презентация

На Windows достаточно дважды щёлкнуть [`START_GUI.cmd`](START_GUI.cmd).
Откроется настольная панель, в которой можно без терминала:

- выбрать датасет и каталог результатов;
- запустить или пересобрать Docker inference;
- проверить итоговые артефакты;
- просматривать query, top-10, confidence и решения об отказе;
- открыть презентацию, README или каталог с результатами.

Новый прогон никогда не перезаписывает существующий release-bundle: приложение
автоматически создаёт отдельный каталог `outputs/runs/run_<timestamp>`.

Браузерный result explorer также показывает реальные `speed`-артефакты,
включая top-10 и решение об отказе:

```powershell
.\.venv\Scripts\python.exe -m prototype.server `
  --dataset-dir "D:\data\falcon-tech" `
  --output-dir outputs\score_optimized_speed --check
.\.venv\Scripts\python.exe -m prototype.server `
  --dataset-dir "D:\data\falcon-tech" `
  --output-dir outputs\score_optimized_speed --port 8080
```

Финальная презентация находится в
[`presentation/Falcon_Tech_Vehicle_ReID.pptx`](presentation/Falcon_Tech_Vehicle_ReID.pptx).
Она уже содержит данные команды, контакты, ссылку на репозиторий и честно
подписанный локальный benchmark. Инструкция по проверке — в
[`presentation/README.md`](presentation/README.md).

## Docker

Контейнер основан на закреплённом по digest PyTorch 2.5.1/CUDA 12.1 образе,
совместимом с указанным организаторами драйвером CUDA 12.2. Он содержит только
speed-allowlist из трёх TorchScript backbone и семи
малых reranker-весов. Интернет после сборки не нужен; dataset монтируется
read-only, outputs — отдельно, root filesystem — read-only.

```bash
docker compose build --pull --no-cache
docker compose run --rm reid-api
```

Или обёртками:

```powershell
.\scripts\run_inference.ps1 -Gpu -Build
```

```bash
./scripts/run_inference.sh --gpu --build
```

Сборка и полный offline-прогон Docker проверены. Контейнер запускается без
сети (`network_mode: none`), с read-only root filesystem и GPU reservation.

## Воспроизводимость и правила

- test query обрабатываются независимо;
- общие структуры строятся только по static gallery;
- `camera_id`, CSV order и ground truth test не используются;
- cross-camera junk удаляется только при validation-оценке;
- inference полностью offline;
- все используемые веса перечислены и захешированы в manifest;
- суммарный размер весов значительно меньше ограничения 2 GB;
- конфигурации качества и скорости разделены и не смешивают результаты.

Основные публичные методы: ConvNeXt, OSNet, DINOv2, CLIP, ArcFace, batch-hard
triplet, gallery-side augmentation, LambdaMART и graph reranking. Лицензия кода — [MIT](LICENSE),
а источники и ограничения сторонних компонентов перечислены в
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Проверки разработки

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

Актуальные результаты тестов и Docker-проверки следует брать из последнего
локального прогона и `outputs/score_optimized_speed/inference_manifest.json`.
