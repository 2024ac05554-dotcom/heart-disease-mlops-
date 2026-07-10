import sys
sys.path.append(".")

import os
import joblib
import numpy as np
import pandas as pd
import pytest

MODEL_PATH = "models/model.pkl"

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="models/model.pkl not found — run `python src/train.py` first",
)


@pytest.fixture
def sample_X():
    n = 10
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "age": rng.integers(30, 70, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(100, 180, n),
        "chol": rng.integers(150, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(100, 190, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 4, n),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    })


def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_predicts_binary_labels(sample_X):
    model = joblib.load(MODEL_PATH)
    preds = model.predict(sample_X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_model_probabilities_sum_to_one(sample_X):
    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(sample_X)
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
