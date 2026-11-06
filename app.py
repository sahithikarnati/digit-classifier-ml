"""Streamlit app: evaluate pre-trained digit classifiers on an uploaded test CSV.
Models are loaded from ./model and are NEVER retrained here."""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

st.set_page_config(page_title="Digit Classifier Lab", page_icon="🔢", layout="wide")

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
TARGET_COL = "target"


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


@st.cache_resource
def load_feature_columns():
    return joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision (macro)": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    try:
        metrics["AUC (OvR macro)"] = roc_auc_score(
            y, y_proba, multi_class="ovr", average="macro", labels=model.classes_)
    except ValueError:  # e.g. uploaded file is missing some classes
        metrics["AUC (OvR macro)"] = float("nan")
    return metrics, y_pred

# ------------------------------------------------------------------ UI
st.title("🔢 Handwritten Digit Classifier Lab")
st.caption("Five pre-trained models on the UCI Optical Recognition of Handwritten Digits "
           "dataset (64 pixel features, 10 classes). Upload test data and compare them.")

with st.sidebar:
    st.header("1 · Upload test data")
    uploaded = st.file_uploader("CSV with 64 pixel columns + `target`", type="csv")
    st.markdown("Use the provided `test_data.csv` from the repository.")
    st.header("2 · Choose model")
    model_name = st.selectbox("Classifier", list(MODEL_FILES))
    compare_all = st.checkbox("Also compare all five models", value=True)

if uploaded is None:
    st.info("👈 Upload `test_data.csv` in the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded)
feature_cols = load_feature_columns()
missing = [c for c in feature_cols + [TARGET_COL] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing[:8]}"
             f"{' ...' if len(missing) > 8 else ''}")
    st.stop()

X, y = df[feature_cols], df[TARGET_COL]
st.success(f"Loaded {len(df)} rows · {len(feature_cols)} features · "
           f"{y.nunique()} classes")
with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20), width="stretch")

# ---- selected model
model = load_model(model_name)
metrics, y_pred = evaluate(model, X, y)

st.subheader(f"Evaluation metrics — {model_name}")
cols = st.columns(len(metrics))
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

