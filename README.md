# Palmer Penguins: автоматический ML-пайплайн


Определение вида пингвина (Adelie, Chinstrap или Gentoo) по длине и глубине
клюва, длине ласта и массе тела. Модель — StandardScaler + LogisticRegression.
Проект для PMLDL Assignment 1


![Три вида пингвинов](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/man/figures/lter_penguins.png)



## Данные

[Palmer Penguins](https://github.com/allisonhorst/palmerpenguins): данные Kristen Gorman
и Palmer Station LTER, лицензия данных CC0. Исходный CSV сохраняется в
`data/raw/penguins.csv`; последующие запуски читают локальную копию.
Используем четыре числовых признака, без пола, острова и года.

## Файлы

Структура повторяет разделение этапов из `task.md` внутри пакета `penguins`.
Задание допускает альтернативную логичную иерархию и инструменты.
Airflow не используется, поэтому `services/airflow` не нужен.
Исследование выполнено скриптом, поэтому пустая папка `notebooks` не создаётся.

```text
penguins/
  datasets/                 # загрузка, исследование, препроцессинг
  models/train.py           # обучение и оценка
  deployment/
    api/server.py           # FastAPI
    api/Dockerfile
    app/web.py              # Streamlit
    app/Dockerfile
  assets/                   # фотографии и лицензии
  display.json              # поля формы, диапазоны и названия видов
  config.py                 # пути и параметры обучения
  pipeline.py               # последовательность этапов и расписание
  run_once.py               # один запуск с выводом в терминал
data/raw/                   # исходные данные
data/processed/             # train/test и медианы
models/                     # сохранённая модель
reports/                    # исследование, метрики и журналы
tests/
compose.yaml
requirements.txt
```

Читать реализацию удобно в порядке `datasets/preprocess.py` → `models/train.py`
→ `deployment/api/server.py` → `deployment/app/web.py` → `pipeline.py`.
Отдельные этапы запускаются напрямую из соответствующих пакетов.

Где менять параметры:

- `config.py`: признаки, seed, доля test, коэффициент IQR, число итераций и интервал.
- `display.json`: подписи полей, начальные значения, диапазоны и фотографии видов.
- `deployment/app/web.py`: тексты и расположение элементов интерфейса.
- `compose.yaml`: порты и настройки контейнеров.

`main()` оставлена в многоэтапных скриптах как короткий сценарий выполнения.
API запускает Uvicorn, поэтому отдельная `main()` ему не нужна.

| Файл | Назначение |
|---|---|
| `penguins/datasets/download.py` | Загрузка CSV при его отсутствии |
| `penguins/datasets/explore.py` | Исследование: пропуски, классы, статистика, диаграмма |
| `penguins/datasets/preprocess.py` | Очистка, разбиение train/test, обработка выбросов и пропусков |
| `penguins/models/train.py` | Обучение, accuracy и macro F1, сохранение модели |
| `penguins/deployment/api/server.py` | FastAPI: `/health`, `/predict`, `/docs` |
| `penguins/deployment/app/web.py` | Streamlit: четыре поля ввода и предсказание через API |
| `penguins/pipeline.py` | Запуск всех этапов и расписание |
| `penguins/run_once.py` | Выполнение одного прогона с выводом в журнал |
| `penguins/config.py` | Пути, признаки, random seed |
| `penguins/deployment/{api,app}/Dockerfile` | Отдельные образы API и приложения |
| `compose.yaml` | Контейнеры, сеть и проверки готовности |
| `tests/test_pipeline.py` | Проверки артефактов и API |

## Запуск

Git Bash на Windows, из корня проекта:

```bash
source .venv/Scripts/activate
python -m penguins.run_once
```

Это разовый прогон с видимым выводом. Для автоматизации:

```bash
python -m penguins.pipeline --schedule
```

В Linux/WSL активируйте окружение через `source .venv/bin/activate`.

Нужны Python 3.12 и запущенный Docker Engine (на Windows — Docker Desktop
с Linux-контейнерами), свободные порты 8000 и 8501.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m penguins.pipeline
```

Linux/macOS: замените `.venv\Scripts\python` на `.venv/bin/python`.
Первый запуск требует интернета для зависимостей, Docker-образов и CSV,
если CSV ещё не добавлен в репозиторий.

- Приложение: http://localhost:8501
- Документация API: http://localhost:8000/docs
- Готовность API: http://localhost:8000/health

Полный пайплайн по расписанию:

```powershell
.venv\Scripts\python -m penguins.pipeline --schedule
```

Первый прогон начинается сразу, следующие — через 300 секунд от начала
предыдущего. Если прогон длится дольше интервала, следующий начнётся через
300 секунд после его завершения. Интервал можно увеличить: `--interval 600`.
При ошибке прогон завершается с ненулевым кодом, планировщик записывает ошибку
и повторяет попытку в следующем цикле. Оставьте терминал открытым; Ctrl+C
останавливает планировщик. Запускайте только один экземпляр планировщика.
Это локальная автоматизация, она не запускается автоматически после перезагрузки ОС.

На каждом прогоне выполняются подготовка, обучение, оценка и
`docker compose up --build -d --force-recreate --wait`.
Новая модель копируется в образ API; контейнеры пересоздаются.
Во время пересоздания возможна краткая недоступность приложения.

Остановить контейнеры:

```powershell
docker compose down
```

## Исследование и отдельные этапы

```powershell
.venv\Scripts\python -m penguins.datasets.download
.venv\Scripts\python -m penguins.datasets.explore
.venv\Scripts\python -m penguins.datasets.preprocess
.venv\Scripts\python -m penguins.models.train
.venv\Scripts\python -m unittest discover -s tests -v
```

Быстрая проверка без Docker: `python -m penguins.pipeline --skip-deploy`.
Этот режим не заменяет полный запуск для сдачи.

В исследовании создаются `reports/eda.json`, `reports/statistics.csv`
и `reports/bill_dimensions.png`. Откройте их перед защитой и объясните
распределение классов, пропуски и различия в размерах клюва.

![Размеры клюва по видам](reports/bill_dimensions.png)

Первый локальный прогон: 344 строки, 275 в train и 69 в test; выбросов
по выбранному правилу в train не обнаружено. Accuracy = 1.0, macro F1 = 1.0.
Это результат одного фиксированного разбиения небольшого датасета,
а не гарантия качества на новых данных. Актуальные результаты — в `reports/metrics.json`.

## Препроцессинг и оценка

Удаляем строки без целевого класса и дубликаты, некорректные числовые значения
превращаем в пропуски. Делим данные 80/20 со стратификацией, seed=42.
Границы выбросов 1.5 IQR вычисляются только по train; выбросы удаляются
только из train. Тестовую выборку не фильтруем по этим границам.
Пропуски заполняем медианами train и сохраняем обработанные CSV.

В артефакт модели включён imputer с теми же медианами, StandardScaler
и LogisticRegression. Масштабирование обучается только на train.
API принимает четыре обязательных положительных конечных числа.
Метрики вычисляются на отложенной тестовой выборке; подбор модели
по этой выборке не выполняется.

Результаты: `models/penguins.joblib`, `reports/metrics.json`, история
`reports/metrics_history.jsonl`. Журналы автоматизации: `reports/pipeline.log`
и `reports/run-*.log`. MLflow/Airflow не используются: условие допускает
другие инструменты, метрики сохраняются в JSON и журналах.

## Пример запроса

POST http://localhost:8000/predict с JSON:

```json
{"bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181, "body_mass_g": 3750}
```

Ответ содержит `species` и `probabilities` для трёх классов.

## Известные ограничения

Они существовали до рефакторинга и не менялись вместе со структурой кода:

- `Ctrl+C` пишет `Scheduler stopped` даже для разового запуска и не возвращает
  специальный код отмены.
- Запускайте один планировщик: между независимыми процессами нет блокировки.
- Диапазоны в `display.json` заданы для текущего CSV, а не вычисляются при обучении.
- UI допускает значения от 0.1, API — любые положительные конечные значения.
  Верхнего жёсткого ограничения нет; API возвращает предупреждения, UI их скрывает.

## Требования задания

Все три этапа запускаются последовательно, включая сборку и запуск двух
Docker-контейнеров. Расписание по умолчанию — 5 минут. Palmer Penguins не входит
в запрещённые датасеты. Метрики записываются в JSON и журналы; альтернативы
MLflow, DVC и Airflow разрешены условием. Для сдачи нужен публичный GitHub-репозиторий
и демонстрация полного автоматического пайплайна с предсказанием через приложение.
