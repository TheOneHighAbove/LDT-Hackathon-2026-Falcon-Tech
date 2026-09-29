# Аудит производительности

## Как считается балл

По ответам организаторов в batch-1 входят JPEG I/O, decode, BBox crop,
preprocessing, forward, postprocessing и L2 normalization. Search и reranking
не входят. Для максимума нужны одновременно медиана batch-1 не более 40 ms и
лучший устойчивый throughput не менее 100 FPS. Скрипт
`scripts/benchmark_release_score_optimized.py` делает 50 warmup, 300
синхронизированных batch-1 запусков и не менее 10 секунд для каждого batch
1/8/16/32.

## Почему ранний результат был лучше

Старое число `36.99 ms / 119.47 FPS` нельзя напрямую сравнивать с финальным:

1. batch-1 усреднялся по одному проходу 96 изображений после 10 warmup, а не
   как медиана 300 отдельных синхронизированных запусков после 50 warmup;
2. замер выполнялся на PyTorch 2.11/CUDA 12.8, тогда как финальный Docker
   использует PyTorch 2.5.1/CUDA 12.1 для гарантированной совместимости с
   заявленным организаторами CUDA 12.2;
3. ранний benchmark не включал пятизонный HSV-признак, который теперь входит в
   реальный `extract()` и используется отказом/reranker.

На одной RTX 4060 и одном строгом скрипте получено:

| Runtime | Batch-1 median | Лучший throughput | Комментарий |
|---|---:|---:|---|
| PyTorch 2.5.1 + CUDA 12.1, финальный Docker | 50.98 ms | 101.04 FPS | профиль сдачи |
| PyTorch 2.11 + CUDA 12.8, эксперимент | 41.83 ms | 124.05 FPS | быстрее, но не гарантирован на драйвере стенда |

Следовательно, примерно 9 ms и 23 FPS объясняются runtime. Остальная разница
с ранними 36.99 ms — протоколом и добавленным preprocessing, а не новым
тяжёлым backbone. Оптимизированный `np.bincount` сохраняет HSV-дескриптор
побитово и убирает около 0.6 ms на локальной машине.

## RTX A5000 против RTX 4060

По официальным спецификациям NVIDIA RTX A5000 имеет 8 192 CUDA-ядра,
27.8 TFLOPS FP32, 24 GB памяти и пропускную способность памяти 768 GB/s.
RTX 4060 имеет 3 072 CUDA-ядра, 15 TFLOPS shader performance и 8 GB памяти.
По пиковому FP32 A5000 примерно в 1.85 раза сильнее, но это не означает
1.85-кратное ускорение всего `extract()`: около половины локального batch-1
уходит на CPU JPEG/crop/HSV, а Xeon Gold 6338 стенда может быть медленнее
i7-12700K в однопоточном preprocessing.

Практический вывод: 100 FPS уже выполнены локально; переход на A5000 должен
сильно ускорить GPU-часть и делает границу 40 ms реалистичной, но заранее
гарантировать 20/20 без запуска на стенде нельзя. На защите нужно показывать
официальный результат организаторов, а локальные числа подписывать моделью
GPU и runtime.

Источники:

- NVIDIA RTX A5000 Data Sheet: https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a5000-datasheet.pdf
- NVIDIA GeForce RTX 4060: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/
- NVIDIA CUDA minor-version compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html

## Команда повторного замера

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_release_score_optimized
```

Перед сравнением нельзя менять веса, конфигурацию, декодер, runtime или
границу измерения. Результат записывается в
`outputs/score_optimized_speed/benchmark.json`.
