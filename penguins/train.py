"""Train a reproducible baseline, evaluate it, and save inference preprocessing."""
import json
from datetime import datetime, timezone
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import PROCESSED, REPORTS, MODEL, FEATURES, TARGET, SEED

def main():
    train = pd.read_csv(PROCESSED / 'train.csv')
    test = pd.read_csv(PROCESSED / 'test.csv')
    metadata = json.loads((PROCESSED / 'preprocessing.json').read_text(encoding='utf-8'))
    # Restore Stage 1 medians in the inference transformer without fitting on test.
    imputer = SimpleImputer(strategy='median').set_output(transform='pandas').fit(pd.DataFrame([metadata['medians']])[FEATURES])
    classifier = Pipeline([('scale', StandardScaler()),
                           ('classifier', LogisticRegression(max_iter=1000, random_state=SEED))])
    classifier.fit(train[FEATURES], train[TARGET])
    model = Pipeline([('impute', imputer), ('model', classifier)])
    predictions = model.predict(test[FEATURES])
    metrics = {'accuracy': accuracy_score(test[TARGET], predictions),
               'macro_f1': f1_score(test[TARGET], predictions, average='macro'),
               'classes': model.classes_.tolist(),
               'confusion_matrix': confusion_matrix(test[TARGET], predictions, labels=model.classes_).tolist(),
               'classification_report': classification_report(test[TARGET], predictions, output_dict=True, zero_division=0),
               'trained_at': datetime.now(timezone.utc).isoformat()}
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    temporary = MODEL.with_suffix('.tmp')
    joblib.dump(model, temporary)
    temporary.replace(MODEL)
    (REPORTS / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    with (REPORTS / 'metrics_history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(metrics) + '\n')
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
