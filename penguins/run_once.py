"""Run all stages with output visible in the terminal."""

import logging
import sys

from penguins.config import LOG_FORMAT
from penguins.pipeline import run


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    run(deploy='--skip-deploy' not in sys.argv)
