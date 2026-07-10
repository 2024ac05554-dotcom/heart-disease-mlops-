import sys
sys.path.append(".")

import pandas as pd
import numpy as np
import pytest

from src.preprocess import build_preprocessor, split_data


@pytest.fixture
def sample_df():
    n = 30
    rng = np.random.default_rng(0)
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
        "target": rng.integers(0, 2, n),
    })


def test_preprocessor_output_has_more_columns_than_input(sample_df):
    X = sample_df.drop("target", axis=1)
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    assert transformed.shape[0] == X.shape[0]
    assert transformed.shape[1] > X.shape[1]


def test_split_data_preserves_class_ratio(sample_df):
    X_train, X_test, y_train, y_test = split_data(sample_df, test_size=0.2)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.15


def test_preprocessor_handles_unseen_category(sample_df):
    X = sample_df.drop("target", axis=1)
    pre = build_preprocessor()
    pre.fit(X)
    unseen_row = X.iloc[[0]].copy()
    unseen_row["thal"] = 99
    result = pre.transform(unseen_row)
    assert result.shape[0] == 1
