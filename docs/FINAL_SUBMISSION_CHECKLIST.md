# Финальная приёмка решения

Дата сверки: 28 сентября 2026 года.

Чек-лист составлен по первичным материалам задачи:

- `7. Фалькон Тех.pdf`;
- `Вопросы и ответы Бизнес 7. Фалькон Тех.xlsx`, лист
  `Объединённые вопросы участников`;
- официальный `evaluate.py`;
- README выданного датасета.

## Обязательная поставка

| Требование | Реализация | Статус |
|---|---|---|
| Одна команда inference | `docker compose run --rm reid-api` | Готово |
| Offline runtime | `network_mode: none`, все веса внутри образа | Готово |
| `submission.csv` | 1110 строк без заголовка, query + 10 уникальных gallery ID | Проверено |
| `embeddings.npy` | `float32`, shape `(1860, 1792)`, query затем gallery | Проверено |
| `candidates.csv` | заголовок `query_id,gallery_id,confidence`, отказ отсутствием строки | Проверено |
| Независимость query | текущее query + заранее построенная static gallery | Проверено по коду |
| Open-set около 20% | identity-disjoint tune/confirmation, 20.13% unknown | Готово |
| Лимит весов | 244 640 363 байта из разрешённого лимита 2 000 000 000 | Проверено |
| Воспроизводимость | закреплённый digest образа, точные версии, SHA-256 весов | Готово |
| Публичность источников | URL, версии и контрольные суммы в `THIRD_PARTY_NOTICES.md` | Готово |

Fail-closed verifier проверяет порядок query, схему файлов, валидность и
уникальность top-10, соответствие принятого кандидата итоговому top-1,
`float32`/finite embeddings, SHA-256 артефактов и весов и общий лимит размера.

## Финальный профиль

Для сдачи используется только профиль `speed`:

- locked validation `mAP@10 = 0.836126`;
- Rank-1 `0.804762`, Rank-5 `0.926279`;
- open-set confirmation: F1 `0.836127`, TNR `0.951613`;
- порог отказа `0.6307946113`;
- локальный benchmark на RTX 3070 Ti Laptop: `51.27 ms` batch-1 и
  `105.48 FPS` batch-32.

Validation и benchmark являются локальными измерениями. Они не выдаются за
результат на закрытом тесте или на RTX A5000 организаторов.

## Повторная техническая проверка

На текущем checkout выполнены:

1. повторная сборка `vehicle-reid:latest` из Dockerfile;
2. полный inference внутри Compose с выключенной сетью и read-only root;
3. повторный verifier на новом result bundle;
4. повторный verifier на заново сгенерированном bundle — `status: ok`; CUDA
   допускает малые численные различия между запусками, поэтому критерием служит
   схема, согласованность SHA-256 внутри отчёта и метрики, а не побайтовое
   совпадение с чужим GPU/runtime;
5. `274 passed` полного набора автоматических тестов;
6. `pip check` — конфликтов зависимостей нет;
7. проверка презентации на отсутствие TODO и незаполненных полей;
8. `git fsck` и `git lfs fsck` — ошибок нет.

## Дополнительные материалы

- `START_GUI.cmd` — desktop-приложение без терминальных команд; при первом
  запуске автоматически создаёт `.venv` и устанавливает GUI-зависимости;
- `prototype.server` — браузерный просмотр зафиксированных результатов;
- `presentation/Falcon_Tech_Vehicle_ReID.pptx` — финальная презентация;
- `docs/ERROR_ANALYSIS.md` — полный error analysis;
- `docs/REFUSAL_POLICY.md` — обоснование open-set policy;
- `docs/SCALING_PLAN.md` — масштабирование static gallery до 1 млн записей.

## Команды сдачи

```bash
docker compose build --pull reid-api
docker compose run --rm --no-deps reid-api
```

Результат появляется в `outputs/score_optimized_speed/`. Локальная проверка:

```powershell
.\.venv\Scripts\python.exe -m src.verify_score_optimized `
  --output-dir outputs/score_optimized_speed
```

## Что проверит организатор

Внутренних незавершённых работ для сдачи нет. Внешними проверками остаются
только запуск на скрытом датасете и официальный performance-замер на RTX A5000;
их результаты невозможно и не следует заявлять заранее.
