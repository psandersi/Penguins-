import json
import unittest
import joblib
import pandas as pd
from fastapi.testclient import TestClient
from penguins.api import app
from penguins.config import FEATURES, MODEL, PROCESSED, REPORTS

class PipelineTests(unittest.TestCase):
    def test_artifacts(self):
        train = pd.read_csv(PROCESSED / 'train.csv')
        test = pd.read_csv(PROCESSED / 'test.csv')
        self.assertFalse(train[FEATURES].isna().any().any())
        self.assertFalse(test[FEATURES].isna().any().any())
        self.assertEqual(set(train.species), {'Adelie', 'Chinstrap', 'Gentoo'})
        self.assertTrue(pd.merge(train, test, on=FEATURES + ['species']).empty)
        model = joblib.load(MODEL)
        metadata = json.loads((PROCESSED / 'preprocessing.json').read_text())
        self.assertEqual(model.named_steps['impute'].statistics_.tolist(), [metadata['medians'][f] for f in FEATURES])
        metrics = json.loads((REPORTS / 'metrics.json').read_text())
        self.assertGreater(metrics['accuracy'], .8)

    def test_api_matches_model_and_validates_input(self):
        example = dict(zip(FEATURES, [39.1, 18.7, 181, 3750]))
        model = joblib.load(MODEL)
        expected = model.predict(pd.DataFrame([example])[FEATURES])[0]
        with TestClient(app) as client:
            self.assertEqual(client.get('/health').status_code, 200)
            response = client.post('/predict', json=example)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['species'], expected)
            self.assertAlmostEqual(sum(response.json()['probabilities'].values()), 1)
            self.assertEqual(client.post('/predict', json={}).status_code, 422)
            self.assertEqual(client.post('/predict', json={**example, 'body_mass_g': -1}).status_code, 422)

if __name__ == '__main__':
    unittest.main()
