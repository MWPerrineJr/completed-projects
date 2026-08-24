"""
Shared data-cleaning / model-fitting logic for the sales dashboard, plus a
CLI entry point that fits the baseline (full, unfiltered) models once and
persists them to disk with joblib.

app.py imports the functions below directly, so there is exactly one
implementation of the cleaning and fitting logic shared between the
interactive app and this offline training script.

Usage:
    python train_models.py [path/to/Sales-Export_2019-2020.csv]

Defaults to Sales-Export_2019-2020.csv next to this script if no path is
given. Writes:
    models/regression_models.joblib   (fitted Linear/Huber/XGBoost pipelines
                                        + held-out predictions/metrics)
    models/timeseries_models.joblib   (fitted manual-ARIMA and auto_arima
                                        models + forecasts/metrics)
    models/metadata.json              (when it was trained, on how many rows)
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

# pmdarima is fragile against newer numpy/scipy builds — degrade gracefully.
try:
    from pmdarima import auto_arima

    AUTO_ARIMA_AVAILABLE = True
except Exception:
    AUTO_ARIMA_AVAILABLE = False

HERE = pathlib.Path(__file__).parent
MODELS_DIR = HERE / "models"
DEFAULT_CSV_NAME = "Sales-Export_2019-2020.csv"


# ----------------------------------------------------------------------
# Data loading & cleaning (mirrors the notebook's wrangling cells)
# ----------------------------------------------------------------------
def load_and_clean(source) -> pd.DataFrame:
    """source: a path-like, file-like, or raw bytes of the sales export CSV."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    df = pd.read_csv(source, index_col="date", parse_dates=True)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"order_value_EUR": "sales_EUR", "cost": "cogs"})

    df["sales_EUR"] = (
        df["sales_EUR"].astype(str).str.replace(",", "", regex=False).astype(float)
    )
    df["order_id"] = (
        df["order_id"].astype(str).str.replace("-", "", regex=False).astype(int)
    )

    df = df.sort_index()
    df["net_sales"] = df["sales_EUR"] - df["cogs"]
    df["pct_change"] = df.groupby("country")["net_sales"].pct_change()

    return df[
        [
            "country",
            "sales_EUR",
            "cogs",
            "net_sales",
            "pct_change",
            "category",
            "customer_name",
            "sales_manager",
            "sales_rep",
            "device_type",
            "order_id",
        ]
    ]


# ----------------------------------------------------------------------
# Regression models
# ----------------------------------------------------------------------
def fit_regression_models(df: pd.DataFrame):
    """Fits Linear/Huber/XGBoost on categorical features only. Returns the
    held-out y_test plus, per model, predictions, metrics, and the fitted
    pipeline itself (so a persisted bundle can be reused for new predictions
    later, not just for re-displaying historical metrics)."""
    data = df.copy()
    data = data.drop(columns=["sales_EUR", "cogs", "pct_change", "order_id", "customer_name"])
    data = data.fillna(0)

    X = data.drop(columns=["net_sales"])
    y = data["net_sales"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categoric_features = X.select_dtypes(include=["object", "category", "bool"]).columns

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]),
                numeric_features,
            ),
            (
                "categoric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categoric_features,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    models = {
        "Linear Regression": LinearRegression(),
        "Huber Regression": HuberRegressor(max_iter=1000),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
    }

    for name, regressor in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            "pipeline": pipe,
            "predictions": preds,
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "mse": mean_squared_error(y_test, preds),
        }

    return y_test.reset_index(drop=True), results


# ----------------------------------------------------------------------
# Time series models
# ----------------------------------------------------------------------
def fit_time_series_models(monthly: pd.Series, holdout_months: int = 3):
    monthly_train = monthly.iloc[:-holdout_months]
    monthly_test = monthly.iloc[-holdout_months:]

    out = {"train": monthly_train, "test": monthly_test, "manual": None, "auto": None, "errors": []}

    try:
        seasonal = monthly_train.shape[0] >= 24  # need >~2 seasons for a 12-month seasonal term
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12) if seasonal else (0, 0, 0, 0)
        arima_model = ARIMA(monthly_train, order=order, seasonal_order=seasonal_order)
        arima_results = arima_model.fit()
        forecast = arima_results.forecast(steps=len(monthly_test))
        out["manual"] = {
            "model": arima_results,
            "forecast": forecast,
            "order": order,
            "seasonal_order": seasonal_order,
            "mae": mean_absolute_error(monthly_test, forecast),
            "rmse": mean_squared_error(monthly_test, forecast) ** 0.5,
        }
    except Exception as exc:  # short training windows can make SARIMA fail outright
        out["errors"].append(f"Manual ARIMA failed: {exc}")

    if AUTO_ARIMA_AVAILABLE:
        try:
            auto_model = auto_arima(
                monthly_train,
                start_p=0, start_q=0, max_p=2, max_q=2, d=None,
                start_P=0, start_Q=0, max_P=1, max_Q=1, D=None,
                m=12, seasonal=True, stepwise=True,
                suppress_warnings=True, error_action="ignore",
            )
            preds = auto_model.predict(n_periods=len(monthly_test))
            preds = pd.Series(preds, index=monthly_test.index, name="auto_arima")
            out["auto"] = {
                "model": auto_model,
                "forecast": preds,
                "order": auto_model.order,
                "seasonal_order": auto_model.seasonal_order,
                "mae": mean_absolute_error(monthly_test, preds),
                "rmse": mean_squared_error(monthly_test, preds) ** 0.5,
            }
        except Exception as exc:
            out["errors"].append(f"auto_arima failed: {exc}")

    return out


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def save_models(regression_bundle: dict, ts_bundle: dict, meta_extra: dict | None = None) -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(regression_bundle, MODELS_DIR / "regression_models.joblib")
    joblib.dump(ts_bundle, MODELS_DIR / "timeseries_models.joblib")

    meta = {"trained_at": datetime.now(timezone.utc).isoformat(), **(meta_extra or {})}
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_models() -> dict | None:
    """Returns None if no persisted bundle exists yet."""
    reg_path = MODELS_DIR / "regression_models.joblib"
    ts_path = MODELS_DIR / "timeseries_models.joblib"
    meta_path = MODELS_DIR / "metadata.json"
    if not (reg_path.exists() and ts_path.exists() and meta_path.exists()):
        return None
    return {
        "regression": joblib.load(reg_path),
        "timeseries": joblib.load(ts_path),
        "metadata": json.loads(meta_path.read_text()),
    }


def train_and_persist(df: pd.DataFrame, holdout_months: int = 3) -> dict:
    """Fits everything on the full (unfiltered) dataset and saves it. Shared
    by the CLI below and by app.py's 'Retrain & save baseline' button."""
    y_test, reg_results = fit_regression_models(df)
    monthly = df["net_sales"].resample("MS").sum().asfreq("MS")
    ts_results = fit_time_series_models(monthly, holdout_months=holdout_months)

    meta = save_models(
        {"y_test": y_test, "results": reg_results},
        ts_results,
        meta_extra={
            "n_rows": int(len(df)),
            "date_range": [str(df.index.min().date()), str(df.index.max().date())],
            "holdout_months": holdout_months,
        },
    )
    return {"regression": {"y_test": y_test, "results": reg_results}, "timeseries": ts_results, "metadata": meta}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    csv_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / DEFAULT_CSV_NAME
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    print(f"Loading {csv_path} ...")
    df = load_and_clean(csv_path)
    print(f"{len(df):,} orders loaded ({df.index.min().date()} to {df.index.max().date()}).")

    print("Fitting regression models on the full dataset...")
    print("Fitting time series models (this can take a minute if auto_arima is enabled)...")
    bundle = train_and_persist(df)

    print(f"\nSaved to {MODELS_DIR}/  (trained_at={bundle['metadata']['trained_at']})")
    for name, r in bundle["regression"]["results"].items():
        print(f"  {name}: R²={r['r2']:.4f}  MAE=€{r['mae']:,.0f}")
    if bundle["timeseries"]["manual"]:
        m = bundle["timeseries"]["manual"]
        print(f"  Manual ARIMA {m['order']}{m['seasonal_order']}: MAE=€{m['mae']:,.0f}")
    if bundle["timeseries"]["auto"]:
        a = bundle["timeseries"]["auto"]
        print(f"  auto_arima {a['order']}{a['seasonal_order']}: MAE=€{a['mae']:,.0f}")
    for err in bundle["timeseries"]["errors"]:
        print(f"  WARNING: {err}")


if __name__ == "__main__":
    main()
