from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/penguins.csv'
PROCESSED = ROOT / 'data/processed'
REPORTS = ROOT / 'reports'
MODEL = ROOT / 'models/penguins.joblib'
FEATURES = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
TARGET = 'species'
SEED = 42
