"""
Sales Analysis Dashboard
------------------------
Interactive Streamlit port of the sales.ipynb notebook: EDA, three
net-sales regressors (Linear / Huber / XGBoost), and two monthly
forecasting approaches (manual SARIMA / auto_arima).

All charts use Plotly so hover, zoom, pan, and legend-click filtering
work natively inside Streamlit (st.pyplot() would only give a static
image).

Run with:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from train_models import (
    AUTO_ARIMA_AVAILABLE,
    DEFAULT_CSV_NAME,
    MODELS_DIR,
    fit_regression_models,
    fit_time_series_models,
    load_and_clean as _load_and_clean,
    load_models as load_persisted_models,
    train_and_persist,
)

st.set_page_config(page_title="Sales Analysis Dashboard", layout="wide")

CAT_FEATURES = ["country", "category", "sales_manager", "sales_rep", "device_type"]


# ----------------------------------------------------------------------
# Cached wrappers around the shared logic in train_models.py.
# Keeping the cleaning/fitting code itself in train_models.py means there is
# exactly one implementation shared between this app and the offline CLI
# used to (re)build the persisted baseline models.
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning sales data...")
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    return _load_and_clean(file_bytes)


@st.cache_data(show_spinner="Training regression models...")
def train_regression_models(df: pd.DataFrame):
    return fit_regression_models(df)


@st.cache_data(show_spinner="Fitting time series models...")
def train_time_series_models(monthly: pd.Series, holdout_months: int = 3):
    return fit_time_series_models(monthly, holdout_months)


@st.cache_resource(show_spinner=False)
def load_persisted_bundle(_cache_bust: float):
    """_cache_bust is the models/ folder's mtime, so retraining automatically
    busts this cache without a manual .clear() call scattered elsewhere."""
    return load_persisted_models()


def models_mtime() -> float:
    reg = MODELS_DIR / "regression_models.joblib"
    return reg.stat().st_mtime if reg.exists() else 0.0


def get_source_bytes() -> bytes | None:
    """Prefer a CSV shipped next to app.py; otherwise let the user upload one."""
    import pathlib

    default_path = pathlib.Path(__file__).parent / DEFAULT_CSV_NAME
    if default_path.exists():
        return default_path.read_bytes()

    st.sidebar.warning(f"'{DEFAULT_CSV_NAME}' not found next to app.py.")
    uploaded = st.sidebar.file_uploader("Upload the sales export CSV", type="csv")
    if uploaded is not None:
        return uploaded.getvalue()
    return None


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
st.title("Sales Analysis Dashboard")
st.caption("European product orders, 2019-2020 — EDA, net-sales regressors, and monthly forecasts.")

source_bytes = get_source_bytes()
if source_bytes is None:
    st.info("Upload the sales export CSV in the sidebar to get started.")
    st.stop()

sales_data = load_and_clean(source_bytes)

# ---- Persisted baseline models ----
persisted = load_persisted_bundle(models_mtime())

with st.sidebar.expander("Baseline models", expanded=persisted is None):
    if persisted:
        st.caption(
            f"Persisted baseline trained {persisted['metadata']['trained_at']} "
            f"on {persisted['metadata']['n_rows']:,} orders."
        )
    else:
        st.caption("No persisted baseline yet — every tab retrains live until you save one.")
    if st.button("Retrain & save baseline", help="Fits on the full, unfiltered dataset and overwrites models/."):
        with st.spinner("Fitting on the full dataset and saving to models/..."):
            train_and_persist(sales_data)
        load_persisted_bundle.clear()
        st.rerun()

# ---- Sidebar filters ----
st.sidebar.header("Filters")


def multiselect_all(label, series):
    options = sorted(series.dropna().unique().tolist())
    return st.sidebar.multiselect(label, options, default=options)


sel_country = multiselect_all("Country", sales_data["country"])
sel_category = multiselect_all("Category", sales_data["category"])
sel_manager = multiselect_all("Sales manager", sales_data["sales_manager"])
sel_device = multiselect_all("Device type", sales_data["device_type"])

min_date, max_date = sales_data.index.min().date(), sales_data.index.max().date()
date_range = st.sidebar.slider(
    "Order date range", min_value=min_date, max_value=max_date, value=(min_date, max_date)
)

filtered = sales_data[
    sales_data["country"].isin(sel_country)
    & sales_data["category"].isin(sel_category)
    & sales_data["sales_manager"].isin(sel_manager)
    & sales_data["device_type"].isin(sel_device)
    & (sales_data.index.date >= date_range[0])
    & (sales_data.index.date <= date_range[1])
]

st.sidebar.caption(f"{len(filtered):,} of {len(sales_data):,} orders match the current filters.")

filters_are_default = (
    set(sel_country) == set(sales_data["country"].unique())
    and set(sel_category) == set(sales_data["category"].unique())
    and set(sel_manager) == set(sales_data["sales_manager"].unique())
    and set(sel_device) == set(sales_data["device_type"].unique())
    and date_range == (min_date, max_date)
)

if filtered.empty:
    st.warning("No orders match the current filters. Widen a filter in the sidebar.")
    st.stop()

# ---- KPI row ----
k1, k2, k3, k4 = st.columns(4)
k1.metric("Orders", f"{len(filtered):,}")
k2.metric("Total net sales", f"€{filtered['net_sales'].sum():,.0f}")
k3.metric("Avg net sales / order", f"€{filtered['net_sales'].mean():,.0f}")
margin_pct = (filtered["net_sales"].sum() / filtered["sales_EUR"].sum()) * 100
k4.metric("Blended margin", f"{margin_pct:.1f}%")

tab_eda, tab_regression, tab_forecast = st.tabs(
    ["Overview & EDA", "Regression Models", "Time Series Forecast"]
)

# ----------------------------------------------------------------------
# EDA tab
# ----------------------------------------------------------------------
with tab_eda:
    st.subheader("Distributions")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered, x="sales_EUR", nbins=50, title="Distribution of sales")
        fig.update_layout(xaxis_title="Sales (EUR)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(filtered, x="cogs", nbins=50, title="Distribution of COGS")
        fig.update_layout(xaxis_title="COGS (EUR)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales vs. COGS by country")
    st.caption("Click a country in the legend to isolate it, or double-click to solo it.")
    fig = px.scatter(
        filtered, x="sales_EUR", y="cogs", color="country", opacity=0.6,
        hover_data=["category", "sales_rep", "device_type"],
        title="Sales vs COGS by country",
    )
    fig.update_layout(xaxis_title="Sales (EUR)", yaxis_title="COGS (EUR)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Period-over-period change in net sales, by country")
    fig = px.box(
        filtered.reset_index(), x="country", y="pct_change",
        title="Percentage change in net sales by country",
        points="outliers",
    )
    fig.update_layout(xaxis_title="Country", yaxis_title="Pct change in net sales")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Regression tab
# ----------------------------------------------------------------------
with tab_regression:
    st.subheader("Predicting net sales from categorical attributes only")
    st.caption(
        "country, category, sales_manager, sales_rep, device_type — sales_EUR and cogs are "
        "excluded because they define net_sales directly."
    )

    use_persisted_reg = filters_are_default and persisted is not None

    if not use_persisted_reg and len(filtered) < 50:
        st.warning("Need at least 50 orders in the current filter to get a meaningful train/test split.")
    else:
        if use_persisted_reg:
            y_test = persisted["regression"]["y_test"]
            results = persisted["regression"]["results"]
            st.caption(f"Serving the persisted baseline (trained {persisted['metadata']['trained_at']}).")
        else:
            y_test, results = train_regression_models(filtered)
            reason = "the current filter differs from the full dataset" if persisted else "no persisted baseline has been saved yet"
            st.caption(f"Retraining live — {reason}.")

        metrics_df = pd.DataFrame(
            {
                name: {"R²": r["r2"], "MAE (€)": r["mae"], "MSE": r["mse"]}
                for name, r in results.items()
            }
        ).T
        st.dataframe(metrics_df.style.format({"R²": "{:.4f}", "MAE (€)": "{:,.0f}", "MSE": "{:,.0f}"}))

        model_choice = st.selectbox("Model", list(results.keys()))
        preds = results[model_choice]["predictions"]

        c1, c2 = st.columns(2)
        with c1:
            lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=preds, mode="markers", opacity=0.6, name=model_choice))
            fig.add_trace(
                go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="black"), name="Perfect forecast")
            )
            fig.update_layout(
                title=f"Actual vs predicted — {model_choice}",
                xaxis_title="Actual net sales (€)",
                yaxis_title="Predicted net sales (€)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            forecasts = pd.DataFrame({"actual": y_test.to_numpy()})
            for name, r in results.items():
                forecasts[name] = r["predictions"]
            forecasts = forecasts.sort_values("actual").reset_index(drop=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=forecasts["actual"], mode="lines", name="Actual", line=dict(color="black")))
            for name in results:
                fig.add_trace(go.Scatter(y=forecasts[name], mode="lines", name=name, opacity=0.85))
            fig.update_layout(
                title="Test-set forecasts vs actual (sorted by actual)",
                xaxis_title="Test orders sorted by actual net sales",
                yaxis_title="Net sales (€)",
            )
            st.plotly_chart(fig, use_container_width=True)

        best = metrics_df["R²"].astype(float).idxmax()
        st.info(
            f"Best R² in the current filter: **{best}** ({metrics_df.loc[best, 'R²']:.4f}). "
            "A negative R² means the model loses to simply predicting the mean net sale — "
            "worth flagging rather than hiding."
        )

# ----------------------------------------------------------------------
# Time series tab
# ----------------------------------------------------------------------
with tab_forecast:
    st.subheader("Monthly net sales forecast")
    st.caption("Order dates are irregular, so this resamples to monthly totals before forecasting.")

    monthly = filtered["net_sales"].resample("MS").sum().asfreq("MS")

    holdout = st.slider("Holdout months", min_value=1, max_value=min(6, max(1, len(monthly) - 6)), value=min(3, max(1, len(monthly) - 6)))

    use_persisted_ts = (
        filters_are_default
        and persisted is not None
        and holdout == persisted["metadata"].get("holdout_months", 3)
    )

    if not use_persisted_ts and len(monthly) < holdout + 6:
        st.warning("Not enough months of data in the current filter to fit and evaluate a forecast.")
    else:
        if use_persisted_ts:
            ts_results = persisted["timeseries"]
            st.caption(f"Serving the persisted baseline forecast (trained {persisted['metadata']['trained_at']}).")
        else:
            ts_results = train_time_series_models(monthly, holdout_months=holdout)
            reason = "the current filter or holdout differs from the saved baseline" if persisted else "no persisted baseline has been saved yet"
            st.caption(f"Retraining live — {reason}.")

        for err in ts_results["errors"]:
            st.warning(err)
        if not AUTO_ARIMA_AVAILABLE:
            st.caption("pmdarima isn't installed in this environment, so auto_arima is skipped. "
                       "`pip install pmdarima` to enable it.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_results["train"].index, y=ts_results["train"], name="Train", line=dict(color="steelblue")))
        fig.add_trace(go.Scatter(x=ts_results["test"].index, y=ts_results["test"], name="Actual", line=dict(color="black")))
        if ts_results["manual"]:
            fig.add_trace(
                go.Scatter(
                    x=ts_results["manual"]["forecast"].index, y=ts_results["manual"]["forecast"],
                    name="Manual ARIMA", line=dict(color="darkorange", dash="dash"),
                )
            )
        if ts_results["auto"]:
            fig.add_trace(
                go.Scatter(
                    x=ts_results["auto"]["forecast"].index, y=ts_results["auto"]["forecast"],
                    name="auto_arima", line=dict(color="seagreen", dash="dash"),
                )
            )
        fig.update_layout(title="Monthly net sales: forecast vs actual", xaxis_title="Month", yaxis_title="Net sales (€)")
        st.plotly_chart(fig, use_container_width=True)

        rows = {}
        if ts_results["manual"]:
            m = ts_results["manual"]
            rows["Manual ARIMA"] = {"Order": f"{m['order']}{m['seasonal_order']}", "MAE (€)": m["mae"], "RMSE (€)": m["rmse"]}
        if ts_results["auto"]:
            a = ts_results["auto"]
            rows["auto_arima"] = {"Order": f"{a['order']}{a['seasonal_order']}", "MAE (€)": a["mae"], "RMSE (€)": a["rmse"]}
        if rows:
            st.dataframe(pd.DataFrame(rows).T.style.format({"MAE (€)": "{:,.0f}", "RMSE (€)": "{:,.0f}"}))
