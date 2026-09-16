import json
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from penguins.deployment.api.server import app
from penguins.config import FEATURES, MODEL, PROCESSED, REPORTS
from penguins import pipeline

EXAMPLE = {
    'bill_length_mm': 39.1,
    'bill_depth_mm': 18.7,
    'flipper_length_mm': 181,
    'body_mass_g': 3750,
}


class PipelineTests(unittest.TestCase):
    def test_artifacts(self):
        train = pd.read_csv(PROCESSED / 'train.csv')
        test = pd.read_csv(PROCESSED / 'test.csv')

        self.assertFalse(train[FEATURES].isna().any().any())
        self.assertFalse(test[FEATURES].isna().any().any())
        self.assertEqual(set(train.species), {'Adelie', 'Chinstrap', 'Gentoo'})
        overlap = pd.merge(train, test, on=FEATURES + ['species'])
        self.assertTrue(overlap.empty)

        model = joblib.load(MODEL)
        metadata_path = PROCESSED / 'preprocessing.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        expected_medians = [metadata['medians'][name] for name in FEATURES]
        actual_medians = model.named_steps['impute'].statistics_.tolist()
        self.assertEqual(actual_medians, expected_medians)

        metrics_path = REPORTS / 'metrics.json'
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        self.assertGreater(metrics['accuracy'], 0.8)

    def test_api_matches_model(self):
        model = joblib.load(MODEL)
        frame = pd.DataFrame([EXAMPLE])[FEATURES]
        expected = model.predict(frame)[0]

        with TestClient(app) as client:
            self.assertEqual(client.get('/health').status_code, 200)
            response = client.post('/predict', json=EXAMPLE)

        self.assertEqual(response.status_code, 200)
        prediction = response.json()
        self.assertEqual(prediction['species'], expected)
        self.assertAlmostEqual(sum(prediction['probabilities'].values()), 1)
        self.assertEqual(prediction['warnings'], [])

    def test_api_validates_input(self):
        with TestClient(app) as client:
            unusual = {**EXAMPLE, 'body_mass_g': 100000}
            response = client.post('/predict', json=unusual)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()['warnings'])

            for invalid in [{}, {**EXAMPLE, 'body_mass_g': -1}]:
                with self.subTest(input=invalid):
                    response = client.post('/predict', json=invalid)
                    self.assertEqual(response.status_code, 422)

    def test_failed_stage_prevents_deployment(self):
        failure = pipeline.subprocess.CalledProcessError(1, 'preprocess')
        with patch.object(pipeline.subprocess, 'run') as execute:
            execute.side_effect = [None, None, failure]
            with self.assertRaises(pipeline.subprocess.CalledProcessError):
                pipeline.run()

        self.assertEqual(execute.call_count, 3)
        last_command = execute.call_args.args[0]
        self.assertEqual(last_command[-1], 'penguins.datasets.preprocess')

    def test_schedule_waits_after_success_and_failure(self):
        arguments = pipeline.argparse.Namespace(
            schedule=True, skip_deploy=False, interval=300,
        )
        cases = [(10, 0, 290), (310, 1, 300), (300, 0, 300)]

        for duration, exit_code, expected_delay in cases:
            with self.subTest(duration=duration, exit_code=exit_code):
                with (
                    patch.object(pipeline, 'run_with_log', return_value=exit_code),
                    patch.object(pipeline.time, 'monotonic', side_effect=[0, duration]),
                    patch.object(pipeline.time, 'sleep', side_effect=KeyboardInterrupt) as sleep,
                ):
                    with self.assertRaises(KeyboardInterrupt):
                        pipeline.run_schedule(arguments)
                    sleep.assert_called_once_with(expected_delay)


if __name__ == '__main__':
    unittest.main()
