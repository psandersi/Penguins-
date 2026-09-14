from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .config import MODEL, FEATURES

@asynccontextmanager
async def lifespan(app):
    app.state.model = joblib.load(MODEL)
    yield

app = FastAPI(title='Palmer Penguins API', lifespan=lifespan)

class Measurements(BaseModel):
    bill_length_mm: float = Field(gt=0, allow_inf_nan=False)
    bill_depth_mm: float = Field(gt=0, allow_inf_nan=False)
    flipper_length_mm: float = Field(gt=0, allow_inf_nan=False)
    body_mass_g: float = Field(gt=0, allow_inf_nan=False)

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict(measurements: Measurements):
    frame = pd.DataFrame([measurements.model_dump()])[FEATURES]
    probabilities = app.state.model.predict_proba(frame)[0]
    return {'species': str(app.state.model.classes_[probabilities.argmax()]),
            'probabilities': dict(zip(app.state.model.classes_.tolist(), probabilities.tolist()))}
