"""Describe the dataset without selecting or tuning the model."""

import json

import matplotlib
import pandas as pd

# The pipeline also runs without a desktop or display server.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from penguins.config import FEATURES, RAW, REPORTS, TARGET


def summarize_data(data):
    return {
        'rows': len(data),
        'columns': list(data.columns),
        'missing': data.isna().sum().to_dict(),
        'duplicates': int(data.duplicated().sum()),
        'species': data[TARGET].value_counts().to_dict(),
    }


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

    REPORTS.mkdir(parents=True, exist_ok=True)
    summary_json = json.dumps(summary, indent=2)
    (REPORTS / 'eda.json').write_text(summary_json, encoding='utf-8')
    data[FEATURES].describe().to_csv(REPORTS / 'statistics.csv')
    save_bill_plot(data)

    print(summary_json)


if __name__ == '__main__':
    main()
