"""Train, evaluate and save the model together with its preprocessing."""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from penguins.config import (
    FEATURES, MAX_ITERATIONS, MODEL, PROCESSED, REPORTS, SEED, TARGET,
)


def train_model(train, medians):
    # Restore the exact medians from Stage 1, rather than learning new ones.
    median_row = pd.DataFrame([medians])[FEATURES]
    imputer = SimpleImputer(strategy='median').set_output(transform='pandas')
    imputer.fit(median_row)

    classifier = Pipeline([
        ('scale', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=MAX_ITERATIONS,
            random_state=SEED,
        )),
    ])
    classifier.fit(train[FEATURES], train[TARGET])

    return Pipeline([
        ('impute', imputer),
        ('model', classifier),
    ])


def evaluate_model(model, test):
    predictions = model.predict(test[FEATURES])
    expected = test[TARGET]
    matrix = confusion_matrix(expected, predictions, labels=model.classes_)
    report = classification_report(
        expected, predictions, output_dict=True, zero_division=0,
    )

    return {
        'accuracy': accuracy_score(expected, predictions),
        'macro_f1': f1_score(expected, predictions, average='macro'),
        'classes': model.classes_.tolist(),
        'confusion_matrix': matrix.tolist(),
        'classification_report': report,
        'trained_at': datetime.now(timezone.utc).isoformat(),
    }


def save_results(model, metrics):
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # A failed write must not replace the last complete model.
    temporary = MODEL.with_suffix('.tmp')
    joblib.dump(model, temporary)
    temporary.replace(MODEL)

    metrics_json = json.dumps(metrics, indent=2)
    (REPORTS / 'metrics.json').write_text(metrics_json, encoding='utf-8')
    history_path = REPORTS / 'metrics_history.jsonl'
    with history_path.open('a', encoding='utf-8') as history:
        history.write(json.dumps(metrics) + '\n')

    print(metrics_json)


def main():
    train = pd.read_csv(PROCESSED / 'train.csv')
    test = pd.read_csv(PROCESSED / 'test.csv')
    metadata_path = PROCESSED / 'preprocessing.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))

    model = train_model(train, metadata['medians'])
    metrics = evaluate_model(model, test)
    save_results(model, metrics)


if __name__ == '__main__':
    main()
