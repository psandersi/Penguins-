"""Split first; learn outlier thresholds and missing-value medians on train only."""
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from .config import RAW, PROCESSED, REPORTS, FEATURES, TARGET, SEED

def main():
    data = pd.read_csv(RAW)[FEATURES + [TARGET]].dropna(subset=[TARGET]).drop_duplicates()
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    data[FEATURES] = data[FEATURES].where(data[FEATURES] > 0)
    train, test = train_test_split(data, test_size=.2, stratify=data[TARGET], random_state=SEED)
    q1, q3 = train[FEATURES].quantile(.25), train[FEATURES].quantile(.75)
    iqr = q3 - q1
    outliers = ((train[FEATURES] < q1 - 1.5 * iqr) | (train[FEATURES] > q3 + 1.5 * iqr)).any(axis=1)
    train = train.loc[~outliers].copy()
    imputer = SimpleImputer(strategy='median').fit(train[FEATURES])
    if len(imputer.statistics_) != len(FEATURES) or not np.isfinite(imputer.statistics_).all():
        raise ValueError('Every feature needs at least one valid training value')
    medians = dict(zip(FEATURES, imputer.statistics_.tolist()))
    train[FEATURES] = imputer.transform(train[FEATURES])
    test[FEATURES] = imputer.transform(test[FEATURES])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / 'train.csv', index=False)
    test.to_csv(PROCESSED / 'test.csv', index=False)
    metadata = {'train_rows': len(train), 'test_rows': len(test),
                'train_outliers_removed': int(outliers.sum()), 'medians': medians,
                'seed': SEED, 'test_fraction': .2}
    (PROCESSED / 'preprocessing.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    main()
