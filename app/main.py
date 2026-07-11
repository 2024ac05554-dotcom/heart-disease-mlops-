import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Heart Disease Risk API")

model = joblib.load("models/model.pkl")


class PatientFeatures(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: PatientFeatures):
    X = pd.DataFrame([features.dict()])
    proba = model.predict_proba(X)[0]
    pred = int(proba.argmax())
    confidence = float(proba[pred])
    return {
        "prediction": pred,
        "label": "disease" if pred == 1 else "no_disease",
        "confidence": round(confidence, 4),
    }
