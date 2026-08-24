"""
Shared data-loading / preprocessing / model logic for the churn dashboard,
plus a CLI entry point that trains the baseline ANN once and persists it to
disk so app.py doesn't retrain a neural net on every rerun.

Mirrors the sales dashboard's train_models.py: one implementation shared
between the interactive app and this offline training script.

Usage:
    python train_model.py [path/to/Churn_Modelling.csv]

Defaults to Churn_Modelling.csv next to this script if no path is given.
Writes:
    models/preprocessor.joblib   (fitted ColumnTransformer + StandardScaler)
    models/ann_model.keras       (fitted Keras Sequential network)
    models/eval.joblib           (y_test, predicted probabilities, feature names)
    models/metadata.json         (when it was trained, on how many rows, headline metrics)
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

HERE = pathlib.Path(__file__).parent
MODELS_DIR = HERE / "models"
DEFAULT_CSV_NAME = "churn_model.csv"

# Columns used as model inputs, in the same order the notebook used them
# (X = ds.iloc[:, 3:-1] -> everything between CustomerId/Surname and Exited).
FEATURE_COLS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
TARGET_COL = "Exited"
ID_COLS = ["RowNumber", "CustomerId", "Surname"]


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_and_clean(source) -> pd.DataFrame:
    """source: a path-like, file-like, or raw bytes of Churn_Modelling.csv."""
    import io

    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    df = pd.read_csv(source)
    df.columns = df.columns.str.strip()
    missing = set(ID_COLS + FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")
    return df


# ----------------------------------------------------------------------
# Preprocessing — mirrors the notebook's LabelEncoder(Gender) +
# OneHotEncoder(Geography) + StandardScaler(everything), just expressed as
# a reusable, picklable sklearn Pipeline instead of three manual steps.
# ----------------------------------------------------------------------
def build_preprocessor() -> Pipeline:
    column_transform = ColumnTransformer(
        transformers=[
            ("geography", OneHotEncoder(handle_unknown="ignore"), ["Geography"]),
            # OrdinalEncoder with an explicit category order reproduces
            # LabelEncoder's alphabetical Female=0 / Male=1 mapping.
            ("gender", OrdinalEncoder(categories=[["Female", "Male"]]), ["Gender"]),
        ],
        remainder="passthrough",  # CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
    )
    return Pipeline([
        ("columns", column_transform),
        ("scale", StandardScaler()),
    ])


# ----------------------------------------------------------------------
# ANN — same architecture as the notebook: Dense(6, relu) x2, Dense(1, sigmoid)
# ----------------------------------------------------------------------
def build_ann():
    import tensorflow as tf

    ann = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=6, activation="relu"),
        tf.keras.layers.Dense(units=6, activation="relu"),
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ])
    ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return ann


def fit_churn_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 0,
                     epochs: int = 100, batch_size: int = 32, verbose: int = 0):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    ann = build_ann()
    history = ann.fit(X_train_t, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    y_pred_proba = ann.predict(X_test_t, verbose=0).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_train_loss": float(history.history["loss"][-1]),
    }
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    return {
        "ann": ann,
        "preprocessor": preprocessor,
        "y_test": y_test,
        "y_pred_proba": y_pred_proba,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "metrics": metrics,
    }


def predict_one(bundle: dict, row: dict) -> float:
    """row: dict with keys matching FEATURE_COLS. Returns churn probability."""
    X = pd.DataFrame([row])[FEATURE_COLS]
    X_t = bundle["preprocessor"].transform(X)
    return float(bundle["ann"].predict(X_t, verbose=0).ravel()[0])


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def save_bundle(bundle: dict, meta_extra: dict | None = None) -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    bundle["ann"].save(MODELS_DIR / "ann_model.keras")
    joblib.dump(bundle["preprocessor"], MODELS_DIR / "preprocessor.joblib")
    joblib.dump(
        {
            "y_test": bundle["y_test"],
            "y_pred_proba": bundle["y_pred_proba"],
            "roc_fpr": bundle["roc_fpr"],
            "roc_tpr": bundle["roc_tpr"],
        },
        MODELS_DIR / "eval.joblib",
    )
    meta = {"trained_at": datetime.now(timezone.utc).isoformat(), "metrics": bundle["metrics"], **(meta_extra or {})}
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_bundle() -> dict | None:
    ann_path = MODELS_DIR / "ann_model.keras"
    prep_path = MODELS_DIR / "preprocessor.joblib"
    eval_path = MODELS_DIR / "eval.joblib"
    meta_path = MODELS_DIR / "metadata.json"
    if not (ann_path.exists() and prep_path.exists() and eval_path.exists() and meta_path.exists()):
        return None

    import tensorflow as tf

    evald = joblib.load(eval_path)
    return {
        "ann": tf.keras.models.load_model(ann_path),
        "preprocessor": joblib.load(prep_path),
        **evald,
        "metadata": json.loads(meta_path.read_text()),
    }


def train_and_persist(df: pd.DataFrame, **fit_kwargs) -> dict:
    bundle = fit_churn_model(df, **fit_kwargs)
    meta = save_bundle(
        bundle,
        meta_extra={"n_rows": int(len(df)), "epochs": fit_kwargs.get("epochs", 100)},
    )
    bundle["metadata"] = meta
    return bundle


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    csv_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / DEFAULT_CSV_NAME
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    print(f"Loading {csv_path} ...")
    df = load_and_clean(csv_path)
    print(f"{len(df):,} customers loaded. Churn rate: {df[TARGET_COL].mean():.1%}")

    print("Training ANN (100 epochs)... this can take a minute or two on CPU.")
    bundle = train_and_persist(df, epochs=100, batch_size=32, verbose=0)

    m = bundle["metrics"]
    print(f"\nSaved to {MODELS_DIR}/  (trained_at={bundle['metadata']['trained_at']})")
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  ROC AUC:   {m['roc_auc']:.4f}")
    print(f"  Confusion matrix: {m['confusion_matrix']}")


if __name__ == "__main__":
    main()
