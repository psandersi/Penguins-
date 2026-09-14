"""Sequential full pipeline, with an optional scheduler and durable run logs."""
import argparse
from datetime import datetime, timezone
import logging
import subprocess
import sys
import time
from .config import ROOT, REPORTS

def run(deploy=True):
    for stage in ['download', 'explore', 'preprocess', 'train']:
        logging.info('Starting %s', stage)
        subprocess.run([sys.executable, '-m', f'penguins.{stage}'], cwd=ROOT, check=True)
    if deploy:
        logging.info('Building and deploying API and app')
        subprocess.run(['docker', 'compose', 'up', '--build', '-d', '--force-recreate', '--wait'], cwd=ROOT, check=True)
    logging.info('Pipeline completed successfully')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--schedule', action='store_true')
    parser.add_argument('--interval', type=int, default=300, help='Seconds between run starts; minimum 300')
    parser.add_argument('--skip-deploy', action='store_true', help='Local ML check only, not a full assignment run')
    args = parser.parse_args()
    if args.interval < 300:
        parser.error('--interval must be at least 300 seconds')
    REPORTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(REPORTS / 'pipeline.log', encoding='utf-8')])
    try:
        while True:
            started = time.monotonic()
            log_path = REPORTS / ('run-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.log')
            # Child output goes into one file per run, including failures and metric values.
            command = [sys.executable, '-m', 'penguins.run_once']
            if args.skip_deploy:
                command.append('--skip-deploy')
            with log_path.open('w', encoding='utf-8') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            logging.info('Run exit code %s; output: %s', result.returncode, log_path)
            if not args.schedule:
                raise SystemExit(result.returncode)
            elapsed = time.monotonic() - started
            delay = args.interval - elapsed if elapsed < args.interval else args.interval
            logging.info('Next run in %.0f seconds; runs never overlap', delay)
            time.sleep(delay)
    except KeyboardInterrupt:
        logging.info('Scheduler stopped')

if __name__ == '__main__':
    main()
