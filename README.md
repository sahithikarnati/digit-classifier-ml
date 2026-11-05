# Handwritten Digit Classification — ML Assignment 2

**Course:** Machine Learning · M.Tech (AIML), BITS Pilani WILP
**Live app:** `<PASTE YOUR STREAMLIT COMMUNITY CLOUD LINK HERE>`

## a. Problem statement

Build and compare five classical machine-learning classifiers that recognise handwritten
digits (0–9) from 8×8 pixel images, and expose them through an interactive Streamlit web app.
This is a **10-class, multi-class classification** problem. For every model we compute
Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC) on a
held-out test set, and the deployed app lets a user upload test data, pick a model and view
the metrics, confusion matrix and classification report without retraining.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Optical Recognition of Handwritten Digits (UCI ML Repository), distributed with scikit-learn as `load_digits()` |
| Source | UCI Machine Learning Repository (Alpaydin & Kaynak) |
| Instances | 1,797 (minimum required: 500) |
| Features | 64 numeric pixel-intensity features, `pixel_r_c` of an 8×8 image, values 0–16 (minimum required: 12) |
| Target | `target` — digit class 0–9 (10 classes, roughly balanced, ~180 samples each) |
| Missing values | None |
| Split | 75 % train (1,347) / 25 % test (450), stratified, `random_state=42` |

`test_data.csv` holds the 450 test rows (64 feature columns + `target`) and is the file to upload in the app.

**Preprocessing:** `StandardScaler` is applied inside the scikit-learn `Pipeline` for Logistic
Regression, KNN and SVM (so the scaler is saved with the model). Decision Tree and Gaussian Naive
Bayes use raw pixel values.

**Metric averaging:** as the problem is multi-class, Precision, Recall and F1 are **macro-averaged**;
AUC is **one-vs-rest macro-averaged** using predicted probabilities; MCC is the multi-class generalisation.
