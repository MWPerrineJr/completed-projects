"""
Customer Churn Dashboard
------------------------
Interactive Streamlit port of ann.ipynb: EDA on the bank-churn dataset, an
ANN classifier (same 6-6-1 dense architecture as the notebook) with
persisted weights, and a form to score a single customer.

All charts are Plotly for hover/zoom/pan. The ANN is expensive to train
(100 epochs), so unlike the sales dashboard this app does NOT retrain live
on every filter change — filters only affect the EDA tab. Model Performance
and Predict a Customer always serve the persisted baseline; retraining is
an explicit sidebar action.

Run with:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from train_model import (
    DEFAULT_CSV_NAME,
    FEATURE_COLS,
    MODELS_DIR,
    load_and_clean as _load_and_clean,
    load_bundle as load_persisted_bundle_raw,
    predict_one,
    train_and_persist,
)

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")


# ----------------------------------------------------------------------
# Cached wrappers
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Loading customer data...")
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    return _load_and_clean(file_bytes)


@st.cache_resource(show_spinner=False)
def load_persisted_bundle(_cache_bust: float):
    """_cache_bust is models/metadata.json's mtime, so retraining busts this
    cache automatically without a manual .clear() scattered elsewhere."""
    return load_persisted_bundle_raw()


def models_mtime() -> float:
    meta = MODELS_DIR / "metadata.json"
    return meta.stat().st_mtime if meta.exists() else 0.0


def get_source_bytes() -> bytes | None:
    import pathlib

    default_path = pathlib.Path(__file__).parent / DEFAULT_CSV_NAME
    if default_path.exists():
        return default_path.read_bytes()

    st.sidebar.warning(f"'{DEFAULT_CSV_NAME}' not found next to app.py.")
    uploaded = st.sidebar.file_uploader("Upload the churn CSV", type="csv")
    if uploaded is not None:
        return uploaded.getvalue()
    return None


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
st.title("Customer Churn Dashboard")
st.caption("Bank customer churn — EDA, an ANN classifier, and a single-customer predictor.")

source_bytes = get_source_bytes()
if source_bytes is None:
    st.info("Upload Churn_Modelling.csv in the sidebar to get started.")
    st.stop()

customers = load_and_clean(source_bytes)

# ---- Persisted baseline model ----
persisted = load_persisted_bundle(models_mtime())

with st.sidebar.expander("Baseline model", expanded=persisted is None):
    if persisted:
        m = persisted["metadata"]["metrics"]
        st.caption(
            f"Trained {persisted['metadata']['trained_at']} on "
            f"{persisted['metadata']['n_rows']:,} customers. "
            f"Test accuracy: {m['accuracy']:.1%}"
        )
    else:
        st.caption("No trained model yet. Train one below (100 epochs, ~1-2 min on CPU).")
    if st.button("Train & save baseline", help="Fits the ANN on the full dataset (100 epochs) and overwrites models/."):
        with st.spinner("Training ANN — 100 epochs, this takes a minute..."):
            train_and_persist(customers, epochs=100, batch_size=32, verbose=0)
        load_persisted_bundle.clear()
        st.rerun()

# ---- Sidebar filters (EDA tab only — the model always serves the persisted baseline) ----
st.sidebar.header("EDA filters")
st.sidebar.caption("These only affect the Overview & EDA tab, not the model.")


def multiselect_all(label, series):
    options = sorted(series.dropna().unique().tolist())
    return st.sidebar.multiselect(label, options, default=options)


sel_geo = multiselect_all("Geography", customers["Geography"])
sel_gender = multiselect_all("Gender", customers["Gender"])
age_min, age_max = int(customers["Age"].min()), int(customers["Age"].max())
age_range = st.sidebar.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max))
sel_active = st.sidebar.selectbox("Active member", ["All", "Active only", "Inactive only"])

filtered = customers[
    customers["Geography"].isin(sel_geo)
    & customers["Gender"].isin(sel_gender)
    & customers["Age"].between(age_range[0], age_range[1])
]
if sel_active == "Active only":
    filtered = filtered[filtered["IsActiveMember"] == 1]
elif sel_active == "Inactive only":
    filtered = filtered[filtered["IsActiveMember"] == 0]

st.sidebar.caption(f"{len(filtered):,} of {len(customers):,} customers match the current filters.")

if filtered.empty:
    st.warning("No customers match the current filters. Widen a filter in the sidebar.")
    st.stop()

# ---- KPI row ----
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(filtered):,}")
k2.metric("Churn rate", f"{filtered['Exited'].mean():.1%}")
k3.metric("Avg balance", f"€{filtered['Balance'].mean():,.0f}")
k4.metric("Avg credit score", f"{filtered['CreditScore'].mean():.0f}")

tab_eda, tab_model, tab_predict = st.tabs(
    ["Overview & EDA", "Model Performance", "Predict a Customer"]
)

# ----------------------------------------------------------------------
# EDA tab
# ----------------------------------------------------------------------
with tab_eda:
    st.subheader("Client credit scores")
    fig = px.histogram(filtered, x="CreditScore", nbins=30, title="Distribution of credit scores")
    fig.update_layout(xaxis_title="Credit score", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Estimated salary by gender")
        fig = px.box(filtered, x="Gender", y="EstimatedSalary", color="Gender", title="Salary by gender")
        fig.update_layout(xaxis_title="Gender", yaxis_title="Estimated salary", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Balance by churn status")
        fig = px.bar(
            filtered.groupby("Exited")["Balance"].mean().reset_index(),
            x="Exited", y="Balance", title="Average balance: stayed (0) vs churned (1)",
        )
        fig.update_layout(xaxis_title="Exited", yaxis_title="Average balance")
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn rate by geography")
    churn_by_geo = filtered.groupby("Geography")["Exited"].mean().reset_index()
    fig = px.bar(churn_by_geo, x="Geography", y="Exited", title="Churn rate by geography")
    fig.update_layout(xaxis_title="Geography", yaxis_title="Churn rate", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Model Performance tab
# ----------------------------------------------------------------------
with tab_model:
    st.subheader("ANN classifier performance")
    st.caption(
        "Trained once on the full dataset (sidebar) rather than per filter — retraining a "
        "neural net on every widget tweak would make the app painfully slow. This tab always "
        "shows the persisted baseline's held-out test performance."
    )

    if not persisted:
        st.warning("No trained model yet. Use 'Train & save baseline' in the sidebar first.")
    else:
        y_test = persisted["y_test"]
        y_pred_proba = persisted["y_pred_proba"]

        threshold = st.slider(
            "Classification threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01,
            help="The notebook hardcodes 0.50. Moving this recomputes the confusion matrix and "
                 "metrics instantly (no retraining) — useful for exploring the precision/recall tradeoff.",
        )
        y_pred = (y_pred_proba > threshold).astype(int)

        import numpy as np
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

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
        m5.metric("ROC AUC", f"{persisted['metadata']['metrics']['roc_auc']:.3f}")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Stayed (0)", "Churned (1)"], y=["Stayed (0)", "Churned (1)"],
                title=f"Confusion matrix @ threshold {threshold:.2f}",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=persisted["roc_fpr"], y=persisted["roc_tpr"], mode="lines", name="ROC curve"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random guess"))
            fig.update_layout(
                title=f"ROC curve (AUC = {persisted['metadata']['metrics']['roc_auc']:.3f})",
                xaxis_title="False positive rate", yaxis_title="True positive rate",
            )
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Predict a Customer tab
# ----------------------------------------------------------------------
with tab_predict:
    st.subheader("Score a single customer")
    st.caption("Mirrors the notebook's manual prediction cell, as an interactive form.")

    if not persisted:
        st.warning("No trained model yet. Use 'Train & save baseline' in the sidebar first.")
    else:
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                credit_score = st.number_input("Credit score", 300, 900, 600)
                geography = st.selectbox("Geography", sorted(customers["Geography"].unique()))
                gender = st.selectbox("Gender", sorted(customers["Gender"].unique()))
            with c2:
                age = st.number_input("Age", 18, 100, 40)
                tenure = st.number_input("Tenure (years)", 0, 15, 3)
                balance = st.number_input("Balance", 0.0, 300000.0, 60000.0, step=1000.0)
            with c3:
                num_products = st.number_input("Number of products", 1, 4, 2)
                has_card = st.checkbox("Has credit card", value=True)
                is_active = st.checkbox("Active member", value=True)
                salary = st.number_input("Estimated salary", 0.0, 300000.0, 50000.0, step=1000.0)

            predict_threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01, key="predict_threshold")
            submitted = st.form_submit_button("Predict")

        if submitted:
            row = {
                "CreditScore": credit_score, "Geography": geography, "Gender": gender,
                "Age": age, "Tenure": tenure, "Balance": balance, "NumOfProducts": num_products,
                "HasCrCard": int(has_card), "IsActiveMember": int(is_active), "EstimatedSalary": salary,
            }
            proba = predict_one(persisted, row)
            verdict = "likely to churn" if proba > predict_threshold else "likely to stay"
            st.metric("Churn probability", f"{proba:.1%}")
            st.info(f"At a {predict_threshold:.2f} threshold, this customer is **{verdict}**.")
