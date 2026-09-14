"""Generate descriptive tables and a plot; does not select/tune the model."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from .config import RAW, REPORTS, FEATURES, TARGET

def main():
    data = pd.read_csv(RAW)
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = {'rows': len(data), 'columns': list(data.columns),
               'missing': data.isna().sum().to_dict(),
               'duplicates': int(data.duplicated().sum()),
               'species': data[TARGET].value_counts().to_dict()}
    (REPORTS / 'eda.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    data[FEATURES].describe().to_csv(REPORTS / 'statistics.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    for species, group in data.groupby(TARGET):
        ax.scatter(group.bill_length_mm, group.bill_depth_mm, label=species, alpha=.7)
    ax.set(xlabel='Bill length (mm)', ylabel='Bill depth (mm)', title='Palmer Penguins')
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / 'bill_dimensions.png', dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
