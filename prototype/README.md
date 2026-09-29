# Falcon ReID GUI и result explorer

## Настольное приложение

На Windows дважды щёлкните `START_GUI.cmd` в корне репозитория. Приложение
при первом запуске создаст локальную `.venv`, установит закреплённую версию
Pillow и затем откроется через `pythonw`. Первый запуск требует Python 3.11+
и доступа к PyPI; дальнейшая работа и inference выполняются offline.

Выданный датасет не хранится в Git. В GUI выберите папку, содержащую
`images/`, `test_query.csv` и `test_gallery.csv`. Начальный путь можно задать
до запуска:

```powershell
$env:DATASET_DIR = "D:\data\falcon-tech"
$env:OUTPUT_DIR = "$PWD\outputs\score_optimized_speed"
.\START_GUI.cmd
```

В GUI доступны:

- выбор каталога датасета и готового result bundle;
- запуск финального Docker inference и опциональная пересборка образа;
- встроенный verifier;
- фильтрация принятых запросов и отказов;
- визуальный просмотр query и top-10 static gallery;
- быстрый доступ к презентации, README и каталогу результатов.

GUI работает fail-safe: если выбранная папка уже содержит файлы, новый Docker
прогон сохраняется рядом в `runs/run_<timestamp>`. Текущий финальный bundle не
перезаписывается.

Для запуска непосредственно из Python:

```powershell
.\.venv\Scripts\python.exe -m prototype.desktop_app
```

Настольное приложение использует только входящий в Python Tkinter и Pillow из
`requirements-gui.txt`. Оно запускает финальный inference внутри Docker, а не
в GUI-окружении.

## Браузерный explorer

Прототип отображает реальные артефакты финального `speed` batch inference:
query, top-10 gallery, решение open-set policy и confidence. Он не содержит
отдельной модели и поэтому не может разойтись с `submission.csv`.

```powershell
.\.venv\Scripts\python.exe -m prototype.server `
  --dataset-dir "D:\data\falcon-tech" `
  --output-dir outputs\score_optimized_speed --check
.\.venv\Scripts\python.exe -m prototype.server `
  --dataset-dir "D:\data\falcon-tech" `
  --output-dir outputs\score_optimized_speed --port 8080
```

После запуска интерфейс доступен на `http://127.0.0.1:8080`.

JSON endpoints:

- `GET /api/health`;
- `GET /api/queries`;
- `GET /api/result/{query_id}`;
- `GET /image/{image_id}` — BBox crop из локального датасета.

Это демонстрационный слой над обязательным batch-пайплайном, а не замена
one-command Docker inference. Изображения не копируются в репозиторий и не
отправляются во внешнюю сеть.
