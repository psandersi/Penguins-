from contextlib import asynccontextmanager
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from penguins.config import DISPLAY, FEATURES, MODEL

RANGES = json.loads(DISPLAY.read_text(encoding='utf-8'))['measurements']


class Measurements(BaseModel):
    bill_length_mm: float = Field(gt=0, allow_inf_nan=False)
    bill_depth_mm: float = Field(gt=0, allow_inf_nan=False)
    flipper_length_mm: float = Field(gt=0, allow_inf_nan=False)
    body_mass_g: float = Field(gt=0, allow_inf_nan=False)


def check_ranges(values):
    warnings = []

    for name, limits in RANGES.items():
        value = values[name]
        if limits['min'] <= value <= limits['max']:
            continue

        message = (
            f"{limits['label']}: {value:g} вне диапазона датасета "
            f"{limits['min']:g}–{limits['max']:g}. "
            'Предсказание может быть ненадёжным.'
        )
        warnings.append(message)

    return warnings


@asynccontextmanager
async def lifespan(app):
    app.state.model = joblib.load(MODEL)
    yield


app = FastAPI(title='Palmer Penguins API', lifespan=lifespan)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(measurements: Measurements):
    values = measurements.model_dump()
    warnings = check_ranges(values)
    frame = pd.DataFrame([values])[FEATURES]

    model = app.state.model
    probabilities = model.predict_proba(frame)[0]
    species = str(model.classes_[probabilities.argmax()])
    class_probabilities = dict(zip(
        model.classes_.tolist(), probabilities.tolist(),
    ))

    return {
        'species': species,
        'probabilities': class_probabilities,
        'warnings': warnings,
    }
