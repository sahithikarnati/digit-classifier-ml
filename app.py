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
