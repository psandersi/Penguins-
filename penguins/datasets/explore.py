"""Standalone EDA of the raw CSV: printed findings, tables and saved plots."""

import json

import matplotlib
import numpy as np
import pandas as pd

# The pipeline also runs without a desktop or display server.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from penguins.config import FEATURES, OUTLIER_FACTOR, RAW, REPORTS, TARGET

EDA_DIRECTORY = REPORTS / 'eda'
LABELS = {
    'bill_length_mm': 'Bill length (mm)',
    'bill_depth_mm': 'Bill depth (mm)',
    'flipper_length_mm': 'Flipper length (mm)',
    'body_mass_g': 'Body mass (g)',
}
COLORS = {'Adelie': '#e69f00', 'Chinstrap': '#8e5ea2', 'Gentoo': '#009e73'}


def summarize_data(data):
    return {
        'rows': len(data),
        'columns': list(data.columns),
        'missing': data.isna().sum().to_dict(),
        'duplicates': int(data.duplicated().sum()),
        'species': data[TARGET].value_counts().to_dict(),
    }


def inspect_columns(data):
    return pd.DataFrame({
        'type': data.dtypes.astype(str),
        'non_null': data.notna().sum(),
        'unique': data.nunique(),
        'missing': data.isna().sum(),
        'missing_percent': data.isna().mean() * 100,
    })


def inspect_measurements(data):
    """Count suspicious values without modifying or filtering the source data."""
    results = []

    for feature in FEATURES:
        values = pd.to_numeric(data[feature], errors='coerce')
        finite_values = values[np.isfinite(values)]
        lower_quartile = finite_values.quantile(0.25)
        upper_quartile = finite_values.quantile(0.75)
        spread = upper_quartile - lower_quartile
        lower_bound = lower_quartile - OUTLIER_FACTOR * spread
        upper_bound = upper_quartile + OUTLIER_FACTOR * spread
        outside_bounds = (finite_values < lower_bound) | (finite_values > upper_bound)

        results.append({
            'feature': feature,
            'non_numeric': int((data[feature].notna() & values.isna()).sum()),
            'infinite': int(np.isinf(values).sum()),
            'non_positive': int((finite_values <= 0).sum()),
            'iqr_lower': lower_bound,
            'iqr_upper': upper_bound,
            'iqr_outliers': int(outside_bounds.sum()),
        })

    return pd.DataFrame(results).set_index('feature')


def save_overview_plots(data):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    for column, axis in zip([TARGET, 'island', 'sex', 'year'], axes.flat):
        counts = data[column].fillna('Missing').value_counts()
        bars = axis.bar(counts.index.astype(str), counts.values, color='#4477aa')
        axis.bar_label(bars)
        axis.set(title=column, ylabel='Number of penguins')
        axis.margins(y=0.15)

    figure.tight_layout()
    figure.savefig(EDA_DIRECTORY / 'categories.png', dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    missing = data.isna().sum().sort_values()
    bars = axis.barh(missing.index, missing.values, color='#cc6677')
    axis.bar_label(bars, padding=3)
    axis.set(title='Missing values before preprocessing', xlabel='Count')
    axis.margins(x=0.15)
    figure.tight_layout()
    figure.savefig(EDA_DIRECTORY / 'missing_values.png', dpi=150)
    plt.close(figure)


def save_feature_plots(data):
    groups = list(data.groupby(TARGET))
    histograms, histogram_axes = plt.subplots(2, 2, figsize=(12, 8))
    boxplots, boxplot_axes = plt.subplots(2, 2, figsize=(12, 8))

    for feature, histogram_axis, boxplot_axis in zip(
        FEATURES, histogram_axes.flat, boxplot_axes.flat,
    ):
        bins = np.histogram_bin_edges(data[feature].dropna(), bins=15)
        samples = []
        names = []

        for species, group in groups:
            values = group[feature].dropna()
            histogram_axis.hist(
                values, bins=bins, alpha=0.5,
                label=species, color=COLORS.get(species),
            )
            samples.append(values)
            names.append(species)

        histogram_axis.set(xlabel=LABELS[feature], ylabel='Count')
        histogram_axis.legend()
        boxplot_axis.boxplot(samples)
        boxplot_axis.set_xticks(range(1, len(names) + 1), names)
        boxplot_axis.set(ylabel=LABELS[feature])

    histograms.tight_layout()
    histograms.savefig(EDA_DIRECTORY / 'distributions.png', dpi=150)
    plt.close(histograms)
    boxplots.tight_layout()
    boxplots.savefig(EDA_DIRECTORY / 'boxplots.png', dpi=150)
    plt.close(boxplots)


def save_relationship_plots(data, correlations):
    figure, axis = plt.subplots(figsize=(8, 7))
    heatmap = axis.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
    axis.set_xticks(range(len(FEATURES)), FEATURES, rotation=30, ha='right')
    axis.set_yticks(range(len(FEATURES)), FEATURES)
    axis.set_title('Pearson correlations: all species combined')

    for row in range(len(FEATURES)):
        for column in range(len(FEATURES)):
            axis.text(column, row, f'{correlations.iloc[row, column]:.2f}',
                      ha='center', va='center')

    figure.colorbar(heatmap, ax=axis)
    figure.tight_layout()
    figure.savefig(EDA_DIRECTORY / 'correlations.png', dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for species, group in data.groupby(TARGET):
        axis.scatter(group['flipper_length_mm'], group['body_mass_g'],
                     label=species, alpha=0.7, color=COLORS.get(species))
    axis.set(xlabel=LABELS['flipper_length_mm'], ylabel=LABELS['body_mass_g'])
    axis.legend()
    figure.tight_layout()
    figure.savefig(EDA_DIRECTORY / 'mass_and_flipper.png', dpi=150)
    plt.close(figure)


def make_report(data, columns, quality, statistics, species_statistics, correlations):
    species_counts = data[TARGET].value_counts()
    class_balance = pd.DataFrame({
        'count': species_counts,
        'percent_of_labeled': species_counts / species_counts.sum() * 100,
    })
    lines = [
        'PALMER PENGUINS — ИССЛЕДОВАНИЕ ДО ПРЕПРОЦЕССИНГА',
        f'Источник: {RAW}',
        f'Размер: {len(data)} строк, {len(data.columns)} столбцов.',
        '\n1. ПЕРВЫЕ СТРОКИ',
        data.head().to_string(index=False),
        '\n2. СТОЛБЦЫ, ТИПЫ И ПРОПУСКИ',
        columns.round(2).to_string(),
        f'Полных дубликатов: {int(data.duplicated().sum())}.',
        f'Строк с пропусками в выбранных признаках: {int(data[FEATURES].isna().any(axis=1).sum())}.',
        f'Строк без целевого вида: {int(data[TARGET].isna().sum())}.',
        '\n3. ВИДЫ И БАЛАНС КЛАССОВ',
        class_balance.round(2).to_string(),
        '\nВиды по островам:',
        pd.crosstab(data['island'], data[TARGET]).to_string(),
        '\nВиды по полу (включая пропуски):',
        pd.crosstab(data['sex'].fillna('Missing'), data[TARGET]).to_string(),
        '\nВиды по годам:',
        pd.crosstab(data['year'], data[TARGET]).to_string(),
        '\n4. ЧИСЛОВЫЕ ПРИЗНАКИ',
        statistics.round(2).to_string(),
        '\nСредние, медианы и диапазоны по видам:',
        species_statistics.T.round(2).to_string(),
        '\n5. ПРОВЕРКА ЗНАЧЕНИЙ И КАНДИДАТОВ В ВЫБРОСЫ',
        quality.round(2).to_string(),
        'IQR рассчитан по всему CSV только для исследования. '
        'Кандидат в выброс не обязательно является ошибкой.',
        '\n6. КОРРЕЛЯЦИИ ПИРСОНА',
        correlations.round(3).to_string(),
        'Корреляции рассчитаны по доступным парам значений. '
        'Связь в общей выборке может отражать различия между видами.',
        '\n7. ВЫВОДЫ ДЛЯ ПРЕПРОЦЕССИНГА',
        f'- Исследованы признаки: {", ".join(FEATURES)}.',
        f'- Пропусков в них: {int(data[FEATURES].isna().sum().sum())}; '
        'медианы для заполнения нужно вычислять только на train.',
        '- Классы представлены разным числом наблюдений: '
        'используем стратификацию и смотрим macro F1 вместе с accuracy.',
        '- Масштабы измерений различаются (мм и г): '
        'StandardScaler обучается только на train.',
        f'- Корреляция массы и длины ласта: '
        f'{correlations.loc["body_mass_g", "flipper_length_mm"]:.3f}. '
        'Смотрите также график по видам, чтобы не смешивать межвидовые различия '
        'со связью внутри одного вида.',
        f'- Отмечено {int(quality["iqr_outliers"].sum())} значений вне IQR-границ '
        '(это число значений, а не уникальных строк). '
        'При препроцессинге границы рассчитываются заново только по train.',
        '- Пол, остров и год описаны для понимания выборки; '
        'текущая модель использует только четыре измерения тела.',
        '- EDA не изменяет исходные данные и не обучает модель. '
        'Не используйте полную выборку для подбора модели и её параметров.',
        '\n8. ГДЕ СМОТРЕТЬ ГРАФИКИ',
        f'- {REPORTS / "bill_dimensions.png"}: размеры клюва по видам.',
        f'- {EDA_DIRECTORY / "categories.png"}: виды, острова, пол и год.',
        f'- {EDA_DIRECTORY / "missing_values.png"}: пропуски.',
        f'- {EDA_DIRECTORY / "distributions.png"}: распределения четырёх признаков.',
        f'- {EDA_DIRECTORY / "boxplots.png"}: диапазоны и выбросы по видам.',
        f'- {EDA_DIRECTORY / "correlations.png"}: корреляции признаков.',
        f'- {EDA_DIRECTORY / "mass_and_flipper.png"}: масса и длина ласта.',
    ]
    return '\n'.join(lines)


def save_bill_plot(data):
    figure, axes = plt.subplots(figsize=(8, 5))

    for species, group in data.groupby(TARGET):
        axes.scatter(
            group['bill_length_mm'],
            group['bill_depth_mm'],
            label=species,
            alpha=0.7,
        )

    axes.set(
        xlabel='Bill length (mm)',
        ylabel='Bill depth (mm)',
        title='Palmer Penguins',
    )
    axes.legend()
    figure.tight_layout()
    figure.savefig(REPORTS / 'bill_dimensions.png', dpi=150)
    plt.close(figure)


def main():
    data = pd.read_csv(RAW)
    summary = summarize_data(data)
    columns = inspect_columns(data)
    quality = inspect_measurements(data)
    statistics = data[FEATURES].describe()
    species_statistics = data.groupby(TARGET)[FEATURES].agg(['mean', 'median', 'min', 'max'])
    correlations = data[FEATURES].corr()

    REPORTS.mkdir(parents=True, exist_ok=True)
    EDA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary_json = json.dumps(summary, indent=2)
    (REPORTS / 'eda.json').write_text(summary_json, encoding='utf-8')
    statistics.to_csv(REPORTS / 'statistics.csv')
    columns.to_csv(EDA_DIRECTORY / 'columns.csv')
    quality.to_csv(EDA_DIRECTORY / 'quality.csv')
    species_statistics.to_csv(EDA_DIRECTORY / 'species_statistics.csv')
    correlations.to_csv(EDA_DIRECTORY / 'correlations.csv')

    save_bill_plot(data)
    save_overview_plots(data)
    save_feature_plots(data)
    save_relationship_plots(data, correlations)

    report = make_report(
        data, columns, quality, statistics, species_statistics, correlations,
    )
    (EDA_DIRECTORY / 'report.txt').write_text(report + '\n', encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
