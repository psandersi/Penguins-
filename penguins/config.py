from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/penguins.csv'
PROCESSED = ROOT / 'data/processed'
REPORTS = ROOT / 'reports'
MODEL = ROOT / 'models/penguins.joblib'
DISPLAY = ROOT / 'penguins/display.json'
ASSETS = ROOT / 'penguins/assets'

FEATURES = [
    'bill_length_mm',
    'bill_depth_mm',
    'flipper_length_mm',
    'body_mass_g',
]
TARGET = 'species'
SEED = 42
TEST_FRACTION = 0.2
OUTLIER_FACTOR = 1.5
MAX_ITERATIONS = 1000

RUN_INTERVAL = 300
LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'
