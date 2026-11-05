import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------- data
digits = load_digits(as_frame=True)
X = digits.data            # 64 pixel features (8x8 images, values 0-16)
y = digits.target          # classes 0-9
feature_cols = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

# test_data.csv: all input features + the target column
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)

# ---------------------------------------------------------------- models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                    random_state=RANDOM_STATE))]),
}

def compute_metrics(model, X_eval, y_eval):
    """Same function is used by the Streamlit app (copied there) - multi-class, macro averaged."""
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)
    return {
        "Accuracy": accuracy_score(y_eval, y_pred),
        "AUC": roc_auc_score(y_eval, y_proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_eval, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_eval, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_eval, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_eval, y_pred),
    }


results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = compute_metrics(model, X_test, y_test)
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join(MODEL_DIR, fname))

joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_columns.joblib"))

table = pd.DataFrame(results).T.round(4)
table.index.name = "ML Model Name"
table.to_csv("model_comparison.csv")
print(table.to_markdown())
print(f"\nTrain rows: {len(X_train)}  Test rows: {len(X_test)}  Features: {X.shape[1]}  Total: {len(X)}")
