"""
Shared data logic + offline training/caching for the fraud detection
dashboard, built from credit_card_fraud.ipynb.

WHY THIS IS SEPARATE FROM app.py, AND WHY IT'S DIFFERENT FROM THE OTHER
TWO DASHBOARDS' train_*.py:
    AIML Dataset.csv is ~493MB / 6.36M rows. Loading that into a live
    Streamlit process on every run (or worse, retraining on it live) would
    make the app painfully slow to open and heavy on memory. So this
    script does the expensive, one-time work OFFLINE:
      1. A single chunked pass over the CSV computes exact aggregates
         (type counts, fraud rate by type, frauds per step, a correlation
         matrix, etc.) using running sums instead of holding the whole
         file in memory, plus a reservoir-sampled subset of rows for the
         charts where an exact full-population read isn't needed (log-
         amount histogram, amount-vs-fraud boxplot, top senders/receivers)
         — sampling is called out explicitly wherever it's used.
      2. It trains the same LogisticRegression pipeline as the notebook
         on the full dataset (using only the 6 model columns, so this read
         is ~300MB, not 493MB) and saves it under the SAME filename the
         existing fraud_detection.py already expects
         (fraud_detection_pipeline.pkl), so that script keeps working.
      3. Everything gets cached to small files under cache/ that app.py
         reads instead of the raw CSV.

Usage:
    uv run python train_model.py [path/to/AIML Dataset.csv]

Run this locally (not through Claude) since it needs the real CSV — it
takes a few minutes on 6.36M rows. Re-run it any time the data changes;
app.py always reflects whatever is currently in cache/.
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
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = pathlib.Path(__file__).parent
CACHE_DIR = HERE / "cache"
PIPELINE_PATH = HERE / "fraud_detection_pipeline.pkl"  # matches fraud_detection.py
DEFAULT_CSV_NAME = "AIML Dataset.csv"

FEATURE_COLS = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
NUMERIC_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
TARGET_COL = "isFraud"
CORR_COLS = NUMERIC_COLS + [TARGET_COL]

CHUNK_SIZE = 1_000_000
SAMPLE_FRACTION = 0.05  # ~318k rows out of 6.36M — plenty for histogram/boxplot shape


# ----------------------------------------------------------------------
# Stage 1: one chunked pass -> exact aggregates + a random sample
# ----------------------------------------------------------------------
def build_eda_cache(csv_path: pathlib.Path, rng_seed: int = 42) -> dict:
    rng = np.random.default_rng(rng_seed)

    total_rows = 0
    fraud_counts = pd.Series(dtype="int64")  # isFraud value counts
    flagged_counts = pd.Series(dtype="int64")  # isFlaggedFraud value counts
    type_stats = pd.DataFrame()  # index=type, columns=[count, fraud_sum]
    step_fraud = pd.Series(dtype="int64")  # index=step, value=fraud count
    zero_after_transfer = 0

    n_corr = 0
    sum_vec = np.zeros(len(CORR_COLS))
    sumsq_mat = np.zeros((len(CORR_COLS), len(CORR_COLS)))

    sample_chunks = []

    usecols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
               "oldbalanceDest", "newbalanceDest", "nameOrig", "nameDest",
               "isFraud", "isFlaggedFraud"]

    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=CHUNK_SIZE)
    for chunk in reader:
        total_rows += len(chunk)

        fraud_counts = fraud_counts.add(chunk["isFraud"].value_counts(), fill_value=0)
        flagged_counts = flagged_counts.add(chunk["isFlaggedFraud"].value_counts(), fill_value=0)

        g = chunk.groupby("type")["isFraud"].agg(count="size", fraud_sum="sum")
        type_stats = type_stats.add(g, fill_value=0) if not type_stats.empty else g

        step_g = chunk.groupby("step")["isFraud"].sum()
        step_fraud = step_fraud.add(step_g, fill_value=0)

        zero_after_transfer += int((
            (chunk["oldbalanceOrg"] > 0)
            & (chunk["newbalanceOrig"] == 0)
            & chunk["type"].isin(["TRANSFER", "CASH_OUT"])
        ).sum())

        X = chunk[CORR_COLS].to_numpy(dtype="float64")
        n_corr += len(X)
        sum_vec += X.sum(axis=0)
        sumsq_mat += X.T @ X

        keep = chunk.sample(frac=SAMPLE_FRACTION, random_state=rng.integers(0, 2**31 - 1))
        sample_chunks.append(keep)

    mean = sum_vec / n_corr
    cov = sumsq_mat / n_corr - np.outer(mean, mean)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr_df = pd.DataFrame(corr, index=CORR_COLS, columns=CORR_COLS)

    sample_df = pd.concat(sample_chunks, ignore_index=True)

    return {
        "total_rows": total_rows,
        "fraud_counts": fraud_counts.astype(int).to_dict(),
        "flagged_counts": flagged_counts.astype(int).to_dict(),
        "type_stats": type_stats.assign(fraud_rate=type_stats["fraud_sum"] / type_stats["count"]),
        "step_fraud": step_fraud,
        "zero_after_transfer": zero_after_transfer,
        "correlation": corr_df,
        "sample": sample_df,
    }


# ----------------------------------------------------------------------
# Stage 2: train the model — same architecture as the notebook
# ----------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(drop="first"), ["type"]),
    ], remainder="drop")


def train_model(csv_path: pathlib.Path, test_size: float = 0.30, random_state: int = 1):
    # Only the 6 model columns + target — ~300MB instead of the full 493MB file.
    df = pd.read_csv(csv_path, usecols=FEATURE_COLS + [TARGET_COL])

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipeline = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # Downsample the ROC curve for compact storage / fast plotting.
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)

    return {
        "pipeline": pipeline,
        "y_test": y_test.to_numpy(),
        "y_pred_proba": y_pred_proba,
        "roc_fpr": fpr[idx],
        "roc_tpr": tpr[idx],
        "metrics": metrics,
    }


# ----------------------------------------------------------------------
# Persistence — small cache files app.py reads instead of the raw CSV
# ----------------------------------------------------------------------
def save_everything(eda: dict, model_bundle: dict, csv_path: pathlib.Path) -> dict:
    CACHE_DIR.mkdir(exist_ok=True)

    joblib.dump(model_bundle["pipeline"], PIPELINE_PATH)

    joblib.dump(
        {
            "y_test": model_bundle["y_test"],
            "y_pred_proba": model_bundle["y_pred_proba"],
            "roc_fpr": model_bundle["roc_fpr"],
            "roc_tpr": model_bundle["roc_tpr"],
        },
        CACHE_DIR / "eval.joblib",
    )

    eda["sample"].to_csv(CACHE_DIR / "eda_sample.csv", index=False)
    eda["correlation"].to_csv(CACHE_DIR / "correlation.csv")
    eda["type_stats"].to_csv(CACHE_DIR / "type_stats.csv")
    eda["step_fraud"].rename("fraud_count").rename_axis("step").to_csv(CACHE_DIR / "step_fraud.csv")

    summary = {
        "total_rows": eda["total_rows"],
        "fraud_counts": eda["fraud_counts"],
        "flagged_counts": eda["flagged_counts"],
        "zero_after_transfer": eda["zero_after_transfer"],
        "sample_fraction": SAMPLE_FRACTION,
    }
    (CACHE_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_path),
        "metrics": model_bundle["metrics"],
    }
    (CACHE_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_cache() -> dict | None:
    """Used by app.py. Returns None if train_model.py hasn't been run yet."""
    required = [
        PIPELINE_PATH,
        CACHE_DIR / "eval.joblib",
        CACHE_DIR / "eda_sample.csv",
        CACHE_DIR / "correlation.csv",
        CACHE_DIR / "type_stats.csv",
        CACHE_DIR / "step_fraud.csv",
        CACHE_DIR / "eda_summary.json",
        CACHE_DIR / "metadata.json",
    ]
    if not all(p.exists() for p in required):
        return None

    evald = joblib.load(CACHE_DIR / "eval.joblib")
    return {
        "pipeline": joblib.load(PIPELINE_PATH),
        "y_test": evald["y_test"],
        "y_pred_proba": evald["y_pred_proba"],
        "roc_fpr": evald["roc_fpr"],
        "roc_tpr": evald["roc_tpr"],
        "sample": pd.read_csv(CACHE_DIR / "eda_sample.csv"),
        "correlation": pd.read_csv(CACHE_DIR / "correlation.csv", index_col=0),
        "type_stats": pd.read_csv(CACHE_DIR / "type_stats.csv", index_col=0),
        "step_fraud": pd.read_csv(CACHE_DIR / "step_fraud.csv"),
        "summary": json.loads((CACHE_DIR / "eda_summary.json").read_text()),
        "metadata": json.loads((CACHE_DIR / "metadata.json").read_text()),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    csv_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / DEFAULT_CSV_NAME
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    print(f"Pass 1/2: streaming {csv_path.name} in {CHUNK_SIZE:,}-row chunks for EDA aggregates...")
    eda = build_eda_cache(csv_path)
    print(f"  {eda['total_rows']:,} rows read. Fraud rate: "
          f"{eda['fraud_counts'].get('1', eda['fraud_counts'].get(1, 0)) / eda['total_rows']:.4%}")
    print(f"  Sampled {len(eda['sample']):,} rows ({SAMPLE_FRACTION:.0%}) for histogram/boxplot/top-N charts.")

    print("Pass 2/2: loading the 6 model columns and training LogisticRegression...")
    model_bundle = train_model(csv_path)
    m = model_bundle["metrics"]
    print(f"  Trained on {m['n_train']:,} rows, tested on {m['n_test']:,}.")
    print(f"  Accuracy: {m['accuracy']:.4f}  Precision: {m['precision']:.4f}  "
          f"Recall: {m['recall']:.4f}  F1: {m['f1']:.4f}  ROC AUC: {m['roc_auc']:.4f}")
    print(f"  Confusion matrix: {m['confusion_matrix']}")

    meta = save_everything(eda, model_bundle, csv_path)
    print(f"\nSaved {PIPELINE_PATH.name} and cache/ (trained_at={meta['trained_at']})")


if __name__ == "__main__":
    main()
