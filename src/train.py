"""
Train and evaluate multiple classifiers on the UFC fight dataset.

Usage:
    python src/train.py

Outputs saved to models/ and reports/figures/.
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import build_features, build_preprocessor
from data_loader import load_raw

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_pipelines(preprocessor):
    return {
        "logistic_regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "xgboost": Pipeline([
            ("prep", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                verbosity=0,
            )),
        ]),
    }


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'='*50}")
    print(f"  {name.upper()}")
    print(f"{'='*50}")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  F1 (Red): {report['1']['f1-score']:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Blue wins", "Red wins"]))

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC Curve — {name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"roc_{name}.png", dpi=120)
    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Blue", "Red"], ax=ax
    )
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cm_{name}.png", dpi=120)
    plt.close(fig)

    return {"name": name, "roc_auc": auc, "accuracy": report["accuracy"], "f1": report["1"]["f1-score"]}


def plot_feature_importance(pipeline, name, feature_names):
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return

    top_n = min(20, len(feature_names))
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), importances[idx], align="center")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"feat_imp_{name}.png", dpi=120)
    plt.close(fig)


def shap_analysis(pipeline, X_test, name):
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    X_transformed = prep.transform(X_test)

    try:
        feature_names = prep.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    sample = X_transformed[:200]
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False, max_display=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"shap_{name}.png", dpi=120, bbox_inches="tight")
    plt.close("all")


def cross_validate_all(pipelines, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("\n── 5-Fold Cross-Validation (ROC-AUC) ──")
    cv_results = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = {"mean": scores.mean(), "std": scores.std()}
        print(f"  {name:25s}  {scores.mean():.4f} ± {scores.std():.4f}")
    return cv_results


def main():
    df = load_raw()
    X, y = build_features(df)
    print(f"Feature matrix: {X.shape}  |  Red-win rate: {y.mean():.2%}")

    preprocessor = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipelines = build_pipelines(preprocessor)

    # Cross-validation
    cv_results = cross_validate_all(pipelines, X, y)

    # Hold-out evaluation
    results = []
    for name, pipe in pipelines.items():
        res = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)
        results.append(res)

        # Feature importance
        try:
            prep = pipe.named_steps["prep"]
            prep.fit(X_train, y_train)
            feature_names = list(prep.get_feature_names_out())
        except Exception:
            feature_names = X.columns.tolist()
        plot_feature_importance(pipe, name, feature_names)

        # SHAP for tree models
        if name in ("random_forest", "xgboost"):
            try:
                shap_analysis(pipe, X_test, name)
            except Exception as e:
                print(f"  SHAP skipped for {name}: {e}")

        # Save model
        joblib.dump(pipe, MODELS_DIR / f"{name}.pkl")
        print(f"  Saved → models/{name}.pkl")

    # Summary table
    summary = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\n── Hold-out Test Summary ──")
    print(summary.to_string(index=False))
    summary.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    best = summary.iloc[0]["name"]
    print(f"\nBest model: {best}  (AUC={summary.iloc[0]['roc_auc']:.4f})")
    print("All artefacts saved to models/ and reports/figures/")


if __name__ == "__main__":
    main()
