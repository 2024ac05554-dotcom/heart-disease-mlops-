import os
import sys
sys.path.append(".")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay,
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
    mlflow.set_experiment("heart-disease-risk")

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

    results = {}          # name -> metrics dict (includes cv_roc_auc)
    fitted_models = {}    # name -> actual fitted Pipeline object

    for name, cfg in model_configs.items():
        pipeline = Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("clf", cfg["estimator"]),
        ])
        search = GridSearchCV(pipeline, cfg["param_grid"], cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)

        print(f"\nBest params for {name}: {search.best_params_}")
        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

        metrics = evaluate(name, search.best_estimator_, X_test, y_test)
        metrics["cv_roc_auc"] = search.best_score_

        with mlflow.start_run(run_name=name):
            mlflow.log_params(search.best_params_)
            mlflow.log_param("model_type", name)

            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            fig, ax = plt.subplots(figsize=(5, 5))
            ConfusionMatrixDisplay.from_estimator(search.best_estimator_, X_test, y_test, ax=ax)
            ax.set_title(f"Confusion Matrix - {name}")
            os.makedirs("reports/figures", exist_ok=True)
            cm_path = f"reports/figures/confusion_{name.replace(' ', '_')}.png"
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(cm_path)

            mlflow.sklearn.log_model(search.best_estimator_, name="model")

        results[name] = metrics
        fitted_models[name] = search.best_estimator_   # keep the real fitted pipeline

    # ---- Champion selection: by CV ROC-AUC (more reliable than a single test split) ----
    champion_name = max(results, key=lambda n: results[n]["cv_roc_auc"])
    champion_pipeline = fitted_models[champion_name]

    print(f"\nChampion model: {champion_name} "
          f"(CV ROC-AUC={results[champion_name]['cv_roc_auc']:.4f}, "
          f"Test ROC-AUC={results[champion_name]['roc_auc']:.4f})")

    os.makedirs("models", exist_ok=True)
    joblib.dump(champion_pipeline, "models/model.pkl")
    print("Saved champion pipeline to models/model.pkl")


if __name__ == "__main__":
    main()