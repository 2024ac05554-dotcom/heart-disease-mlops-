import logging
import time

import joblib
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("heart-disease-api")

app = FastAPI(title="Heart Disease Risk API")

model = joblib.load("models/model.pkl")

REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests", ["status"])
REQUEST_LATENCY = Histogram("predict_latency_seconds", "Prediction request latency")


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


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(features: PatientFeatures):
    start_time = time.time()
    try:
        X = pd.DataFrame([features.dict()])
        proba = model.predict_proba(X)[0]
        pred = int(proba.argmax())
        confidence = float(proba[pred])

        logger.info(f"Prediction request: input={features.dict()} -> "
                    f"prediction={pred}, confidence={confidence:.4f}")

        REQUEST_COUNT.labels(status="success").inc()

        return {
            "prediction": pred,
            "label": "disease" if pred == 1 else "no_disease",
            "confidence": round(confidence, 4),
        }
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        logger.exception("Prediction failed")
        raise
    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)
