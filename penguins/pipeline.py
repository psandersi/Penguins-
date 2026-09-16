"""Последовательно запускает весь проект: от исходного CSV до веб-приложения.

Сначала проверяем наличие данных и строим EDA-отчёт с графиками. Затем очищаем
данные, делим их на train/test, обучаем модель и сохраняем метрики. В конце
Docker Compose собирает образы и запускает API и приложение в двух контейнерах.
Если этап завершился с ошибкой, следующие этапы в этом прогоне не запускаются.

По умолчанию выполняется один прогон. С --schedule повторяем его каждые 5 минут.
Если прогон занял весь интервал или больше, ждём ещё один интервал после него.
Прогоны одного планировщика не пересекаются; после ошибки он повторит попытку
в следующем цикле. Вывод каждого прогона сохраняется в отдельный файл в reports.
Флаг --skip-deploy позволяет проверить данные и обучение без запуска Docker.
"""

import argparse
from datetime import datetime, timezone
import logging
import subprocess
import sys
import time

from penguins.config import LOG_FORMAT, REPORTS, ROOT, RUN_INTERVAL

STAGES = [
    'penguins.datasets.download',
    'penguins.datasets.explore',
    'penguins.datasets.preprocess',
    'penguins.models.train',
]
DEPLOY_COMMAND = [
    'docker', 'compose', 'up', '--build', '-d', '--force-recreate', '--wait',
]


def run(deploy=True):
    for stage in STAGES:
        logging.info('Starting %s', stage)
        command = [sys.executable, '-m', stage]
        subprocess.run(command, cwd=ROOT, check=True)

    if deploy:
        logging.info('Building and deploying API and app')
        subprocess.run(DEPLOY_COMMAND, cwd=ROOT, check=True)

    logging.info('Pipeline completed successfully')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--schedule', action='store_true')
    parser.add_argument(
        '--interval',
        type=int,
        default=RUN_INTERVAL,
        help='Seconds between run starts; minimum 300',
    )
    parser.add_argument(
        '--skip-deploy',
        action='store_true',
        help='Local ML check only, not a full assignment run',
    )
    arguments = parser.parse_args()

    if arguments.interval < RUN_INTERVAL:
        parser.error('--interval must be at least 300 seconds')

    return arguments


def setup_logging():
    REPORTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(REPORTS / 'pipeline.log', encoding='utf-8'),
        ],
    )


def run_with_log(skip_deploy):
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log_path = REPORTS / f'run-{timestamp}.log'
    command = [sys.executable, '-m', 'penguins.run_once']
    if skip_deploy:
        command.append('--skip-deploy')

    # Capture child output separately so every run has its own complete history.
    with log_path.open('w', encoding='utf-8') as log_file:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    logging.info('Run exit code %s; output: %s', result.returncode, log_path)
    return result.returncode


def run_schedule(arguments):
    while True:
        started = time.monotonic()
        exit_code = run_with_log(arguments.skip_deploy)
        if not arguments.schedule:
            raise SystemExit(exit_code)

        elapsed = time.monotonic() - started
        if elapsed < arguments.interval:
            delay = arguments.interval - elapsed
        else:
            delay = arguments.interval

        logging.info('Next run in %.0f seconds; runs never overlap', delay)
        time.sleep(delay)


def main():
    arguments = parse_arguments()
    setup_logging()

    try:
        run_schedule(arguments)
    except KeyboardInterrupt:
        logging.info('Scheduler stopped')


if __name__ == '__main__':
    main()
