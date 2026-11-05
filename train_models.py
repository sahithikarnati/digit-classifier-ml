"""Train five classifiers on the UCI Optical Recognition of Handwritten Digits dataset
(as shipped with scikit-learn), evaluate them, and save everything the Streamlit app needs.

Run:  python train_models.py
"""
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
