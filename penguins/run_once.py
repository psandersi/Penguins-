import logging
import sys
from .pipeline import run

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(deploy='--skip-deploy' not in sys.argv)
