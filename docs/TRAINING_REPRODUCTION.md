# Воспроизведение обучения и релизного стека

Этот документ разделяет два разных сценария:

- проверка сдаваемого решения — полностью воспроизводима из Git и вложенных
  весов одной Docker-командой;
- повторное обучение с нуля — исследовательский конвейер, требующий исходного
  `train.csv`, изображений, публичных pretrained-весов и промежуточных cache.

Релизный inference не скачивает модели и не зависит от training cache.

## 1. Окружение и данные

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
```

Положите `train.csv`, `test_query.csv`, `test_gallery.csv` и `images/` в
`dataset/`. Публичные pretrained-веса загружаются только при подготовке и
обучении; их источники перечислены в `THIRD_PARTY_NOTICES.md`.

## 2. Зафиксированный identity-disjoint split

```powershell
.\.venv\Scripts\python.exe -m src.split `
  --input dataset\train.csv --output-dir splits `
  --val-fraction 0.20 --seed 42 --min-val-images 2 --min-val-cameras 2
```

Ожидаемый результат для выданного train: 7 646 строк train и 1 910 строк val,
1 233 и 308 identity соответственно. Все дальнейшие команды запускаются из
корня репозитория и используют `splits/train.csv`, `splits/val.csv`.

## 3. Backbone и базовые cache

Порядок зависимостей:

```text
split
├── OSNet loss-branch training ──> OSNet train/val embeddings + parts
├── DINOv2 target adaptation ────> CLS/part/patch train/val cache
└── ConvNeXt extraction ─────────> global/part train/val cache
                                      │
                                      └── fused validation descriptors
```

Команды основных обучаемых веток:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_osnet_loss_branches
.\.venv\Scripts\python.exe -m scripts.train_dinov2_vehicle_metric
```

Скрипты extraction в `scripts/extract_*.py` создают cache в
`outputs/expert_fusion/cache/`. Перед следующим этапом должны существовать как
минимум OSNet train/val embeddings и parts, DINOv2 CLS/parts/patches train/val,
ConvNeXt train/val embeddings и parts. Каждый cache хранит `image_id`; загрузка
fail-closed отклоняет несовпадающий порядок.

## 4. Reranker в точном порядке

После создания cache модели обучаются в такой последовательности:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_dino_patch_matcher
.\.venv\Scripts\python.exe -m scripts.train_dino_token_cross_top50
.\.venv\Scripts\python.exe -m scripts.train_dino_top25_verifier
.\.venv\Scripts\python.exe -m scripts.train_strict_family_gnn
.\.venv\Scripts\python.exe -m scripts.train_family_lambdarank
.\.venv\Scripts\python.exe -m scripts.train_modern_gallery_linker
```

Итоговый speed-allowlist — `dino_token_cross_top50.pt`,
`strict_family_gnn_smoothap.pt`, `dino_top25_verifier.pt`,
`family_lambdarank_appearance_lbs.joblib`, два DINO patch matcher и
`modern_gallery_linker_lossbranch.joblib`. Названия и SHA-256 окончательно
проверяет release manifest.

## 5. Экспорт production backbone

```powershell
.\.venv\Scripts\python.exe -m scripts.export_score_optimized_backbones
```

Экспорт создаёт TorchScript в `weights/release/`. Для speed-профиля используются
ровно `convnext_global_parts.ts`, `osnet_loss_branch_global_parts.ts` и
`dinov2_vehicle_cls_tokens.ts`; CLIP остаётся только в quality-профиле.

## 6. Закрытая confirmation и отказ

Сначала собирается точный production descriptor, затем валидируется замороженный
ranking и только после этого калибруется отказ:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_score_optimized_val_embeddings `
  --release-config configs\score_optimized_speed.json
.\.venv\Scripts\python.exe -m scripts.validate_release_profile `
  --preprocessing pillow_shared --dino-size 224 `
  --output outputs\score_optimized_speed\validation.json
.\.venv\Scripts\python.exe -m scripts.calibrate_score_optimized_refusal `
  --release-config configs\score_optimized_speed.json `
  --annotations splits\val.csv --images-dir dataset\images `
  --weights-dir weights `
  --output outputs\score_optimized_speed\refusal_calibration.json
```

Tune seeds обучают логистическую модель и выбирают порог; пять disjoint
confirmation seed используются только один раз для отчёта. Значения из отчётов
переносятся в `configs/score_optimized_speed.json` только после заморозки.

## 7. Финальная приёмка

```powershell
docker compose build --pull reid-api
docker compose run --rm --no-deps reid-api
.\.venv\Scripts\python.exe -m src.verify_score_optimized `
  --output-dir outputs\score_optimized_speed
.\.venv\Scripts\python.exe -m pytest -q
```

Версии в JSON — локальная identity-disjoint validation, а не скрытый test score.
Release-сборка считается воспроизведённой, если verifier принимает схему,
порядок query, hashes и лимит весов; побайтовое совпадение CUDA-выходов нужно
требовать только на одинаковых GPU, драйвере и runtime.
