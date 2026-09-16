"""Подготовка train/test без использования тестовых данных для настройки очистки."""

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from penguins.config import (
    FEATURES,
    OUTLIER_FACTOR,
    PROCESSED,
    RAW,
    REPORTS,
    SEED,
    TARGET,
    TEST_FRACTION,
)


def load_data():
    data = pd.read_csv(RAW)[FEATURES + [TARGET]]
    # Без известного вида нельзя обучаться; повторы придают одной записи лишний вес.
    data = data.dropna(subset=[TARGET]).drop_duplicates()

    # Ошибочные измерения считаем пропусками: позже заполним их медианами train.
    measurements = data[FEATURES].apply(pd.to_numeric, errors='coerce')
    measurements = measurements.replace([np.inf, -np.inf], np.nan)
    data[FEATURES] = measurements.where(measurements > 0)
    return data


def remove_outliers(train):
    # Правило 1.5 IQR — простой способ найти крайние значения без предположения
    # о нормальном распределении. Это правило очистки, а не пределы биологии.
    lower_quartile = train[FEATURES].quantile(0.25)
    upper_quartile = train[FEATURES].quantile(0.75)
    spread = upper_quartile - lower_quartile

    lower_bound = lower_quartile - OUTLIER_FACTOR * spread
    upper_bound = upper_quartile + OUTLIER_FACTOR * spread
    outside_bounds = (train[FEATURES] < lower_bound) | (train[FEATURES] > upper_bound)
    outliers = outside_bounds.any(axis=1)

    return train.loc[~outliers].copy(), int(outliers.sum())


def fill_missing_values(train, test):
    # Медиана меньше среднего зависит от крайних значений. Учим её только на train,
    # чтобы информация из test не участвовала в подготовке модели.
    imputer = SimpleImputer(strategy='median')
    imputer.fit(train[FEATURES])

    # Полностью пустой признак заполнить медианой невозможно.
    if len(imputer.statistics_) != len(FEATURES):
        raise ValueError('Every feature needs at least one valid training value')
    if not np.isfinite(imputer.statistics_).all():
        raise ValueError('Every feature needs at least one valid training value')

    train[FEATURES] = imputer.transform(train[FEATURES])
    test[FEATURES] = imputer.transform(test[FEATURES])
    return dict(zip(FEATURES, imputer.statistics_.tolist()))


def save_data(train, test, medians, outlier_count):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / 'train.csv', index=False)
    test.to_csv(PROCESSED / 'test.csv', index=False)

    # Сохраняем медианы, чтобы модель при предсказании заполняла пропуски так же.
    metadata = {
        'train_rows': len(train),
        'test_rows': len(test),
        'train_outliers_removed': outlier_count,
        'medians': medians,
        'seed': SEED,
        'test_fraction': TEST_FRACTION,
    }
    metadata_json = json.dumps(metadata, indent=2)
    metadata_path = PROCESSED / 'preprocessing.json'
    metadata_path.write_text(metadata_json, encoding='utf-8')
    print(metadata_json)


def main():
    data = load_data()
    # 20% оставляем для оценки. Стратификация сохраняет доли видов,
    # а фиксированный seed позволяет повторить то же разбиение.
    train, test = train_test_split(
        data,
        test_size=TEST_FRACTION,
        stratify=data[TARGET],
        random_state=SEED,
    )

    # Удаляем выбросы только из train: test должен включать и сложные наблюдения,
    # иначе оценка качества получится неоправданно высокой.
    train, outlier_count = remove_outliers(train)
    medians = fill_missing_values(train, test)
    save_data(train, test, medians, outlier_count)


if __name__ == '__main__':
    main()
