# Final-score optimization audit — 2026-09-24

## Решение

Подготовлены два честно разделённых релизных профиля:

| Профиль | Locked mAP@10 | Rank-1 | Rank-5 | Вес |
|---|---:|---:|---:|---:|
| quality | 0.841086 | 0.812875 | 0.936684 | 422 931 269 B |
| speed | 0.836126 | 0.804762 | 0.926279 | 244 640 363 B |

Для максимизации общего score контейнер использует `speed`: CLIP даёт только
около `+0.0053 mAP`, но заметно ухудшает
latency, throughput и VRAM. Quality-профиль остаётся воспроизводимым отдельной
конфигурацией и не смешивается с performance-цифрами speed-профиля.

## Официальные составляющие

- ranking: 45 баллов;
- latency и throughput: 10 + 10;
- инженерная часть: 15;
- open-set отказ: 10;
- защита: 10 (в эту работу не входит).

Полный performance score требует batch-1 `extract()` ≤40 ms и лучшего
throughput ≥100 FPS. В `extract()` входят JPEG I/O, decode, crop, transforms,
forward, postprocessing и L2 normalization; gallery search/reranking не входят.

## Производительность

После удаления flip-TTA и DINOv3, экспорта self-contained TorchScript и
параллельного запуска независимых backbone через CUDA streams получен следующий
замер speed-профиля на NVIDIA GeForce RTX 3070 Ti Laptop GPU:

| Batch-1 median | Лучший throughput | Batch | Peak CUDA |
|---:|---:|---:|---:|
| 51.27 ms | **105.48 FPS** | 32 | 555.2 MiB |

Протокол соответствует Q&A: 50 warmup, 300 синхронизированных batch-1
измерений с медианой и ≥10 секунд для каждого batch 1/8/16/32. В замер входят
исходные JPEG, BBox crop, HSV descriptor, preprocessing, transfer, forward и
pooling. Это локальный аппаратно-зависимый результат, а не обещание баллов на
официальной A5000.

## Open-set

Refusal head откалиброван на фактический top-1 финального reranker. Каждый
эпизод содержит один query на identity, 20.13% unknown query и gallery из 750
изображений. Порог `0.630795` выбран только на tune group-OOF при ограничении
`TNR >= 0.94`, после чего политика один раз проверена на confirmation seed:

- F1: 0.836127;
- TNR: 0.951613;
- `0.7 × F1 + 0.3 × TNR`: 0.870773.

На тех же confirmation эпизодах предыдущая release-политика давала 0.862576;
изменение повышает open-set компонент на 0.008196 и не влияет на ranking.
Калибровка quality хранится отдельно в его конфигурации.

## Engineering acceptance

- one-command local и Docker inference;
- explicit offline weight allowlist;
- 2 GB fail-closed check и SHA-256 всех весов/артефактов;
- независимость от других query, camera ID и CSV order;
- детерминированный stable top-10;
- schema/top-1/finite-value verifier;
- release-код изолирован от training/probe модулей;
- Docker build, offline inference и verifier реально выполнены локально.

## Команды

```powershell
.\.venv\Scripts\python.exe -m scripts.infer_score_optimized `
  --release-config configs/score_optimized_speed.json `
  --output-dir outputs/score_optimized_speed

.\.venv\Scripts\python.exe -m src.verify_score_optimized `
  --output-dir outputs/score_optimized_speed

.\.venv\Scripts\python.exe -m scripts.benchmark_release_score_optimized
```

## Передача следующему разработчику

### Что уже сделано

- основной релизный профиль `speed` зафиксирован в
  `configs/score_optimized_speed.json`: locked mAP@10 `0.836126`, Rank-1
  `0.804762`, Rank-5 `0.926279`;
- сохранён отдельный профиль `quality` с locked mAP@10 `0.841086`, чтобы
  улучшения качества можно было продолжать без смешивания с быстрым релизом;
- тяжёлые DINOv3, CLIP и flip-TTA удалены из speed-пути; три extractor-ветки
  экспортированы в self-contained TorchScript и запускаются через CUDA streams;
- strict end-to-end benchmark включает JPEG I/O, decode, BBox crop,
  preprocessing, transfer, forward, pooling и L2 normalization;
- официальный локальный протокол на RTX 3070 Ti Laptop показал `51.27 ms`
  batch-1 median и `105.48 FPS` при batch-32;
- open-set threshold откалиброван под 20.13% unknown и production descriptor:
  F1 `0.836127`, TNR `0.951613`, итог отказа `0.870773`;
- подготовлены offline Dockerfile/Compose, one-command wrappers, weight
  allowlist, SHA-256 manifest, лимит весов 2 GB и fail-closed verifier;
- проверены стабильный top-10, схема результатов, finite values, соответствие
  top-1, независимость от camera ID, порядка CSV и других query;
- чистый Docker build и полный offline inference успешно выполнены; итоговые
  `submission.csv`, `embeddings.npy` и `candidates.csv` воспроизводятся
  побайтово относительно зафиксированного speed-релиза.

### Финальный статус

Все внутренние пункты подготовки закрыты 27 сентября 2026 года:

1. speed inference и verifier повторно запущены в собранном Docker-образе;
2. `submission.csv`, `candidates.csv`, `embeddings.npy` и `gallery_index.npz`
   воспроизводятся побайтово;
3. схемы, top-10, формат отказа, SHA-256 и размер весов проверены;
4. полный набор тестов и `pip check` проходят;
5. финальная презентация и рабочее desktop-приложение входят в поставку.

Официальный performance-замер на RTX A5000 и запуск на скрытом датасете
выполняются организаторами. Локальные цифры по-прежнему явно подписываются как
результат RTX 3070 Ti Laptop и не выдаются за официальный score. Полная матрица
готовности находится в `docs/FINAL_SUBMISSION_CHECKLIST.md`.
