# Handwritten Digit Classification — ML Assignment 2

**Course:** Machine Learning · M.Tech (AIML), BITS Pilani WILP
**Live app:** https://digit-classifier-ml.streamlit.app/

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

## c. GitHub repository link
https://github.com/sahithikarnati/digit-classifier-ml
```
project-folder/
│-- app.py                 # Streamlit app (loads saved models, never retrains)
│-- train_models.py        # trains + evaluates the 5 models, writes model/ and test_data.csv
│-- requirements.txt
│-- README.md
│-- test_data.csv          # 450 held-out rows (64 features + target)
│-- model_comparison.csv   # metrics table produced by train_models.py
│-- model/                 # logistic_regression, decision_tree, knn, naive_bayes,
│                          # support_vector_machine (.joblib) + feature_columns.joblib
```

Reproduce: `pip install -r requirements.txt && python train_models.py && streamlit run app.py`

## d. Models used

### Comparison table (450-row test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9778 | 0.9991 | 0.9778 | 0.9777 | 0.9776 | 0.9753 |
| Decision Tree | 0.8200 | 0.9008 | 0.8197 | 0.8190 | 0.8187 | 0.8001 |
| KNN | 0.9644 | 0.9922 | 0.9651 | 0.9642 | 0.9642 | 0.9606 |
| Naive Bayes | 0.8289 | 0.9630 | 0.8555 | 0.8284 | 0.8305 | 0.8128 |
| Support Vector Machine | 0.9800 | 0.9992 | 0.9804 | 0.9798 | 0.9799 | 0.9778 |

Hyper-parameters: Logistic Regression (`max_iter=2000`), Decision Tree (`max_depth=12`),
KNN (`k=5`), Gaussian Naive Bayes (defaults), SVM (RBF kernel, `C=10`, `gamma="scale"`, `probability=True`).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong (accuracy 0.978, AUC 0.999) for a linear model. Standardised pixel intensities are close to linearly separable between most digit pairs, and the model is fast, stable and well calibrated. Only marginally behind SVM. |
| Decision Tree | Weakest model (accuracy 0.820, AUC 0.901). Axis-aligned splits on individual pixels do not capture the spatial stroke patterns that define digits, and a single deep tree overfits the 1,347 training images. A noticeably lower AUC reflects poorly ranked class probabilities from sparse leaf estimates. |
| KNN | Accurate (0.964) because similar digits lie close together in pixel space. Slightly below Logistic Regression and SVM, and it stores the full training set, so prediction is slower and the saved model file is the largest. |
| Naive Bayes | Accuracy of only 0.829, though AUC is high (0.963). The assumption of independent, Gaussian-distributed pixels is badly violated as neighbouring pixels are strongly correlated, so hard predictions are often wrong even though class probabilities still rank reasonably. Its precision (0.856) is noticeably above its recall (0.828). |
| Support Vector Machine | Best overall (accuracy 0.980, AUC 0.9992, MCC 0.978). The RBF kernel models non-linear boundaries in the 64-dimensional pixel space while the margin objective keeps it from overfitting. |
| **Overall winner for this dataset** | **Support Vector Machine (RBF)** is best on every metric, narrowly ahead of Logistic Regression (a difference of about 1 test image out of 450, so the two are practically comparable). SVM is recommended; Logistic Regression is the simpler, near-equivalent alternative. |

## Streamlit app features

- CSV upload of test data (must contain all 64 feature columns and `target`)
- Model selection dropdown (5 models)
- Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix heat-map and classification report
- Optional side-by-side comparison of all five models on the uploaded data
- Saved models are loaded from `model/`; nothing is retrained on the uploaded data
