# Palmer Penguins: автоматический ML-пайплайн

Определение вида пингвина (Adelie, Chinstrap или Gentoo) по длине и глубине
клюва, длине ласта и массе тела. Модель — StandardScaler + LogisticRegression.
Проект для PMLDL Assignment 1: подготовка данных → обучение и оценка → Docker API и приложение.

![Три вида пингвинов](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/man/figures/lter_penguins.png)

Artwork by @allison_horst. [Источник и разрешение на учебное использование](https://allisonhorst.github.io/palmerpenguins/articles/art.html).
Изображение загружается с GitHub, поэтому для его отображения нужен интернет.

## Данные

[Palmer Penguins](https://github.com/allisonhorst/palmerpenguins): данные Kristen Gorman
и Palmer Station LTER, лицензия данных CC0. Исходный CSV сохраняется в
`data/raw/penguins.csv`; последующие запуски читают локальную копию.
Используем четыре числовых признака, без пола, острова и года.

## Файлы

| Файл | Назначение |
|---|---|
| `penguins/download.py` | Загрузка CSV при его отсутствии |
| `penguins/explore.py` | Исследование: пропуски, классы, статистика, диаграмма |
| `penguins/preprocess.py` | Очистка, разбиение train/test, обработка выбросов и пропусков |
| `penguins/train.py` | Обучение, accuracy и macro F1, сохранение модели |
| `penguins/api.py` | FastAPI: `/health`, `/predict`, `/docs` |
| `penguins/app.py` | Streamlit: четыре поля ввода и предсказание через API |
| `penguins/pipeline.py` | Запуск всех этапов и расписание |
| `penguins/run_once.py` | Выполнение одного прогона с выводом в журнал |
| `penguins/config.py` | Пути, признаки, random seed |
| `deployment/*.Dockerfile` | Отдельные образы API и приложения |
| `compose.yaml` | Контейнеры, сеть и проверки готовности |
| `tests/test_pipeline.py` | Проверки артефактов и API |

## Запуск

Нужны Python 3.12 и запущенный Docker Engine (на Windows — Docker Desktop
с Linux-контейнерами), свободные порты 8000 и 8501. Команды выполняются
из папки `assign1` (или корня репозитория, если опубликовано её содержимое).

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
.venv\Scripts\python -m penguins.download
.venv\Scripts\python -m penguins.explore
.venv\Scripts\python -m penguins.preprocess
.venv\Scripts\python -m penguins.train
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

## Сдача

Опубликуйте содержимое проекта в публичном GitHub-репозитории и сдайте ссылку.
Модель и обработанные данные исключены из Git: они генерируются пайплайном.
На защите покажите два последовательных запуска по расписанию, журналы
с метриками, два работающих контейнера (`docker compose ps`) и предсказание
через веб-приложение. Все три этапа и автоматизация обязательны.
