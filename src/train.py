import sys
sys.path.append(".")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from src.preprocess import load_clean_data, build_preprocessor, split_data


def evaluate(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")
    return metrics


def main():
    df = load_clean_data("data/heart_clean.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    model_configs = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {"clf__C": [0.01, 0.1, 1, 10]},
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "param_grid": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 5, 10],
            },
        },
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, cfg in model_configs.items():
        pipeline = Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("clf", cfg["estimator"]),
        ])
        search = GridSearchCV(pipeline, cfg["param_grid"], cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)

        print(f"\nBest params for {name}: {search.best_params_}")
        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

        results[name] = evaluate(name, search.best_estimator_, X_test, y_test)


if __name__ == "__main__":
    main()