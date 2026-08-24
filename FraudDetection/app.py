"""
Fraud Detection Dashboard
-------------------------
Interactive Streamlit port of credit_card_fraud.ipynb, extending the
existing fraud_detection.py (which still works standalone) into a full
EDA + model performance + prediction dashboard.

IMPORTANT — this app never reads AIML Dataset.csv directly. At 493MB /
6.36M rows, loading that live would make the app slow to open and heavy
on memory. Instead it reads the small files under cache/ that
train_model.py produces offline. If cache/ doesn't exist yet, run:

    uv run python train_model.py

once locally (a few minutes on the full dataset), then start the app:

    uv run streamlit run app.py

Charts marked "sampled" are drawn from a 5% random sample of the data
(collected by train_model.py) rather than the full 6.36M rows — plenty
for histogram/boxplot/top-N shapes without holding the whole file in
memory. Charts NOT marked sampled (type counts, fraud rate by type,
frauds over time, the correlation matrix, zero-balance-after-transfer)
are computed exactly, from a full streaming pass over every row.
"""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from train_model import load_cache

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


@st.cache_resource(show_spinner="Loading cached model and EDA data...")
def get_cache(_cache_bust: float):
    return load_cache()


def cache_mtime() -> float:
    import pathlib

    meta = pathlib.Path(__file__).parent / "cache" / "metadata.json"
    return meta.stat().st_mtime if meta.exists() else 0.0


def fraud_count(counts: dict, key: int) -> int:
    """JSON round-trips dict keys as strings; handle both."""
    return int(counts.get(str(key), counts.get(key, 0)))


st.title("Fraud Detection Dashboard")
st.caption("Mobile-money transaction fraud — EDA on 6.36M transactions, a logistic regression classifier, and a transaction scorer.")

cache = get_cache(cache_mtime())

if cache is None:
    st.error(
        "No cached model/EDA data found. This app reads pre-computed results from cache/ "
        "rather than loading the 493MB source CSV directly."
    )
    st.code("uv run python train_model.py", language="bash")
    st.caption("Run that once locally (it needs AIML Dataset.csv in this folder), then reload this page.")
    st.stop()

summary = cache["summary"]
meta = cache["metadata"]
total_rows = summary["total_rows"]
fraud_rows = fraud_count(summary["fraud_counts"], 1)
flagged_rows = fraud_count(summary["flagged_counts"], 1)

with st.sidebar.expander("Cache status", expanded=False):
    st.caption(f"Built {meta['trained_at']} from {meta['source_csv']}")
    st.caption(f"{total_rows:,} rows aggregated exactly; charts marked 'sampled' use {summary['sample_fraction']:.0%} of rows.")
    st.caption("To refresh: run `uv run python train_model.py` locally, then reload this page.")

st.sidebar.header("EDA filter")
type_options = sorted(cache["type_stats"].index.tolist())
sel_types = st.sidebar.multiselect("Transaction type", type_options, default=type_options)
st.sidebar.caption("Applies to the Overview & EDA tab only — the exact-aggregate charts (frauds over time, correlation) always reflect the full dataset.")

# ---- KPI row (exact, whole-dataset) ----
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total transactions", f"{total_rows:,}")
k2.metric("Fraud rate", f"{fraud_rows / total_rows:.3%}")
k3.metric("Fraudulent transactions", f"{fraud_rows:,}")
k4.metric("Flagged by the source system", f"{flagged_rows:,}")

tab_eda, tab_model, tab_predict = st.tabs(["Overview & EDA", "Model Performance", "Predict a Transaction"])

# ----------------------------------------------------------------------
# EDA tab
# ----------------------------------------------------------------------
with tab_eda:
    type_stats = cache["type_stats"].loc[cache["type_stats"].index.isin(sel_types)]
    sample = cache["sample"][cache["sample"]["type"].isin(sel_types)]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Transaction types")
        st.caption("Exact — full dataset.")
        fig = px.bar(type_stats.reset_index(), x="type", y="count", title="Transaction volume by type")
        fig.update_layout(xaxis_title="Type", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Fraud rate by type")
        st.caption("Exact — full dataset.")
        fig = px.bar(
            type_stats.reset_index().sort_values("fraud_rate", ascending=False),
            x="type", y="fraud_rate", title="Fraud rate by transaction type",
        )
        fig.update_layout(xaxis_title="Type", yaxis_title="Fraud rate", yaxis_tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Transaction amount distribution (log scale)")
        st.caption(f"Sampled — {summary['sample_fraction']:.0%} of rows.")
        fig = px.histogram(sample, x=np.log1p(sample["amount"]), nbins=100, title="log(amount + 1)")
        fig.update_layout(xaxis_title="Log(amount + 1)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.subheader("Amount vs. fraud (under €50k)")
        st.caption(f"Sampled — {summary['sample_fraction']:.0%} of rows.")
        under_50k = sample[sample["amount"] < 50000]
        fig = px.box(under_50k, x="isFraud", y="amount", title="Amount by fraud status")
        fig.update_layout(xaxis_title="isFraud", yaxis_title="Amount")
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Frauds over time (by step)")
    st.caption("Exact — full dataset. Not affected by the type filter above.")
    step_fraud = cache["step_fraud"]
    fig = go.Figure(go.Scatter(x=step_fraud["step"], y=step_fraud["fraud_count"], mode="lines"))
    fig.update_layout(title="Fraud count per time step", xaxis_title="Step (hour)", yaxis_title="Fraud count")
    st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Correlation matrix")
        st.caption("Exact — full dataset.")
        corr = cache["correlation"]
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Feature correlation")
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        st.subheader("Zero balance after transfer/cash-out")
        st.caption("Exact — full dataset. A classic fraud signature: sender's balance goes to exactly zero.")
        st.metric("Matching transactions", f"{summary['zero_after_transfer']:,}")
        st.metric("Share of all transactions", f"{summary['zero_after_transfer'] / total_rows:.3%}")

    st.subheader("Most active accounts")
    st.caption(f"Estimated from the {summary['sample_fraction']:.0%} sample, not an exact full-dataset count.")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.write("Top senders")
        st.dataframe(sample["nameOrig"].value_counts().head(10).rename("transactions"))
    with t2:
        st.write("Top receivers")
        st.dataframe(sample["nameDest"].value_counts().head(10).rename("transactions"))
    with t3:
        st.write("Top senders of fraud")
        st.dataframe(sample[sample["isFraud"] == 1]["nameOrig"].value_counts().head(10).rename("frauds sent"))

# ----------------------------------------------------------------------
# Model Performance tab
# ----------------------------------------------------------------------
with tab_model:
    st.subheader("Logistic regression classifier")
    st.caption(
        f"Trained on {meta['metrics']['n_train']:,} transactions, evaluated on a held-out "
        f"{meta['metrics']['n_test']:,} ({meta['trained_at']}). Retraining happens offline "
        "(`uv run python train_model.py`) — this tab always shows that run's results."
    )

    y_test = cache["y_test"]
    y_pred_proba = cache["y_pred_proba"]

    threshold = st.slider(
        "Classification threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01,
        help="The notebook uses the default 0.50 cutoff. Moving this recomputes the confusion "
             "matrix and metrics instantly from the stored test-set probabilities — no retraining.",
    )
    y_pred = (y_pred_proba > threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.1%}")
    m2.metric("Precision", f"{prec:.1%}")
    m3.metric("Recall", f"{rec:.1%}")
    m4.metric("F1", f"{f1:.3f}")
    m5.metric("ROC AUC", f"{meta['metrics']['roc_auc']:.3f}")

    st.caption(
        "class_weight='balanced' trades accuracy for recall on the rare fraud class — with "
        "fraud at roughly 0.1% of transactions, a model that always predicts 'legit' would still "
        "score >99% accuracy while catching zero fraud. Precision/recall/F1 are the metrics that "
        "actually reflect performance here, not accuracy alone."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Legit (0)", "Fraud (1)"], y=["Legit (0)", "Fraud (1)"],
            title=f"Confusion matrix @ threshold {threshold:.2f}",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cache["roc_fpr"], y=cache["roc_tpr"], mode="lines", name="ROC curve"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random guess"))
        fig.update_layout(
            title=f"ROC curve (AUC = {meta['metrics']['roc_auc']:.3f})",
            xaxis_title="False positive rate", yaxis_title="True positive rate",
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Predict a Transaction tab
# ----------------------------------------------------------------------
with tab_predict:
    st.subheader("Score a single transaction")
    st.caption("Same fields as fraud_detection.py's standalone form, now inside the full dashboard.")

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            transaction_type = st.selectbox("Transaction type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT", "CASH_IN"])
            amount = st.number_input("Amount", min_value=0.0, value=1000.0)
            old_orig = st.number_input("Old balance (sender)", min_value=0.0, value=10000.0)
        with c2:
            new_orig = st.number_input("New balance (sender)", min_value=0.0, value=9000.0)
            old_dest = st.number_input("Old balance (receiver)", min_value=0.0, value=0.0)
            new_dest = st.number_input("New balance (receiver)", min_value=0.0, value=0.0)

        predict_threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01, key="predict_threshold")
        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([{
            "type": transaction_type, "amount": amount,
            "oldbalanceOrg": old_orig, "newbalanceOrig": new_orig,
            "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        }])
        proba = float(cache["pipeline"].predict_proba(row)[0, 1])
        st.metric("Fraud probability", f"{proba:.2%}")
        if proba > predict_threshold:
            st.error(f"At a {predict_threshold:.2f} threshold, this transaction looks like fraud.")
        else:
            st.success(f"At a {predict_threshold:.2f} threshold, this transaction looks legitimate.")
