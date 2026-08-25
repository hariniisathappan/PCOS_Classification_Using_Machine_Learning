# PCOS Classification Using Machine Learning

**A Systematic Study of Feature Engineering, Feature Selection, and Bayesian Hyperparameter Optimization for Polycystic Ovary Syndrome (PCOS) Prediction**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Models](https://img.shields.io/badge/models-14%20classifiers-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Overview

Polycystic Ovary Syndrome (**PCOS**) is the most common endocrine disorder in women of reproductive age, affecting **8–20%** of this population worldwide. Delayed or missed diagnosis is linked to infertility, type 2 diabetes, and cardiovascular disease.

This repository documents an end-to-end ML pipeline for PCOS prediction on a **541-patient clinical dataset (41 raw features)**, evaluating **14 classifiers across 9 algorithm families** under **4 experimental conditions**: Baseline vs. Bayesian-Tuned, each With vs. Without Feature Engineering.

---

## 🗂️ Notebook Pipeline (Actual Execution Order)

```mermaid
flowchart TD
    N1[01_data_exploration.ipynb<br/>EDA on 541-patient, 44-feature dataset] --> N2

    N2[02_preprocessing_pipeline.ipynb<br/>Imputation, Outlier Capping, Encoding,<br/>Scaling, Class-weighting] --> N7

    N7[07_FE_and_FS.ipynb<br/>Rotterdam-based Feature Engineering<br/>+ 3-Method Voting Feature Selection<br/>41 → 11 features] --> N3
    N7 --> N4
    N2 --> N5
    N2 --> N6

    N5[05_without_FE_baseline.ipynb<br/>14 models, default hyperparams,<br/>41 raw features] --> N8
    N3[03_with_FE_Baseline.ipynb<br/>14 models, default hyperparams,<br/>11 selected features] --> N8

    N6[06_without_FE_Tuned.ipynb<br/>14 models, Optuna TPE tuning,<br/>41 raw features] --> N8
    N4[04_with_FE_Tuned.ipynb<br/>14 models, Optuna TPE tuning,<br/>11 selected features] --> N8

    N8[08_Time_and_performance.ipynb<br/>Training time & optimization<br/>time comparison] --> N9
    N9[09_Trainind_data_metrics.ipynb<br/>Full metric tables:<br/>Accuracy, Precision, Recall, F1, AUC-ROC] --> N10
    N10[10_Learning_curve.ipynb<br/>Learning curves: train vs test,<br/>overfitting/generalization checks] --> FINAL

    FINAL[Final Comparative Analysis:<br/>4-Condition Benchmark + Four Archetype Framework]
```

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Exploratory Data Analysis — class balance, distributions, missing values, correlations |
| `02_preprocessing_pipeline.ipynb` | Unified cleaning pipeline: imputation, outlier capping, encoding, scaling, class weighting |
| `03_with_FE_Baseline.ipynb` | 14 classifiers, **default hyperparameters**, **11 engineered/selected features** |
| `04_with_FE_Tuned.ipynb` | 14 classifiers, **Optuna Bayesian tuning**, **11 engineered/selected features** |
| `05_without_FE_baseline.ipynb` | 14 classifiers, **default hyperparameters**, **41 raw features** |
| `06_without_FE_Tuned.ipynb` | 14 classifiers, **Optuna Bayesian tuning**, **41 raw features** |
| `07_FE_and_FS.ipynb` | Rotterdam-criteria feature engineering (7 new features) + 3-method voting feature selection (41 → 11) |
| `08_Time_and_performance.ipynb` | Training/optimization time benchmarking across all 4 conditions |
| `09_Trainind_data_metrics.ipynb` | Consolidated metrics tables for all 14 models × 4 conditions |
| `10_Learning_curve.ipynb` | Learning curve analysis — training data size vs. F1/AUC-ROC, overfitting diagnostics |

---

## 🎯 Key Highlights

| Aspect | Detail |
|---|---|
| Dataset size | 541 patients (364 negative, 177 positive) — 67.3% / 32.7% class split |
| **Raw features** | **41** clinical, hormonal, anthropometric & ultrasound variables |
| Engineered features | 7 clinically derived (Rotterdam-based) |
| Final selected features | 11 (10 raw + 1 engineered, via 3-method voting) |
| EPV improvement | 4.31 → 16.09 (satisfies EPV ≥ 10 clinical validity threshold) |
| Models evaluated | 14 classifiers across 9 algorithm families |
| Cross-validation | Stratified 10-fold |
| **Best Baseline Model** | AdaBoost (F1 = 0.845, with FE) |
| **Best Baseline Recall** | Logistic Regression (Recall = 88.7%, with FE) |
| **Best Tuned Model (overall)** | SVM (RBF kernel) — F1 = 0.8807, Recall = 90.6% |
| **Best Calibrated Model** | Logistic Regression — Brier Score = 0.069 |

---

## 📁 Repository Structure
PCOS_Classification_Using_ML/
│
├── data/ # Raw and processed datasets
├── models/ # Saved trained model artifacts (.joblib)
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_preprocessing_pipeline.ipynb
│ ├── 03_with_FE_Baseline.ipynb
│ ├── 04_with_FE_Tuned.ipynb
│ ├── 05_without_FE_baseline.ipynb
│ ├── 06_without_FE_Tuned.ipynb
│ ├── 07_FE_and_FS.ipynb
│ ├── 08_Time_and_performance.ipynb
│ ├── 09_Trainind_data_metrics.ipynb
│ └── 10_Learning_curve.ipynb
├── requirements.txt
├── .gitignore
└── README.md

*(`outputs/` is generated locally when you run the notebooks — see .gitignore)*

---

## 🧬 Dataset Description & Exploratory Analysis (Notebook 01)

- **Source:** Publicly available PCOS clinical dataset (Kaggle)
- **Patients:** 541 total (177 PCOS-positive, 364 PCOS-negative)
- **Raw features:** 41, spanning demographic, anthropometric, hormonal, metabolic, menstrual-history, ultrasound, and lifestyle variables.

<p align="center">
  <img src="https://github.com/user-attachments/assets/784de37a-0f48-4d6f-891d-4e22bfdb653f" width="850" alt="PCOS Target Distribution and Class Percentage"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/9fde2582-2448-48ad-a84b-a19b02a80171" width="850" alt="Correlation Heatmap of all 41 features"/>
</p>

**Follicle count** (right and left ovary) was consistently the strongest visual separator between PCOS-positive and PCOS-negative patients:

<p align="center">
  <img src="https://github.com/user-attachments/assets/54dd9602-2dea-481c-bf8d-c48db2c19a65" width="850" alt="Follicle No. Right - Distribution and Boxplot by PCOS Status"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/8e256225-794b-485d-a23e-3263eecde62e" width="850" alt="Follicle No. Left - Distribution and Boxplot by PCOS Status"/>
</p>

---

## ⚙️ Data Pre-processing Pipeline (Notebook 02)

Applied identically across **all four experimental conditions** to ensure fair comparison:

1. **Missing values:** Median imputation (continuous), Mode imputation (categorical)
2. **Outliers:** IQR-based capping at 1.5× IQR boundaries
3. **Categorical encoding:** One-hot encoding (sklearn conventions)
4. **Scaling:** Standardization (zero mean, unit variance)
5. **Class imbalance:** `class_weight="balanced"` + Stratified 10-fold CV (SMOTE intentionally avoided — moderate 2:1 imbalance + leakage risk)

---

## 🧪 Feature Engineering & Selection (Notebook 07)

### Rotterdam-Criteria-Based Engineered Features

| # | Engineered Feature | Clinical Rationale |
|---|---|---|
| 1 | **LH/FSH Ratio** | Elevated ratio is a known PCOS hormonal biomarker |
| 2 | **AMH × BMI (interaction)** | Joint ovarian reserve + metabolic status signal |
| 3 | **Follicle Asymmetry** | \|Left − Right\| follicle count captures unilateral dysregulation |
| 4 | **Cycle Regularity Indicator** | Encodes Rotterdam's oligo-anovulation criterion |
| 5 | **Hyperandrogenism Composite** | Weighted androgen marker combination |
| 6 | **Insulin Resistance Proxy** | Fasting insulin × fasting glucose (HOMA-IR style) |
| 7 | **Endometrial Thickness / Cycle Length Ratio** | Reproducibility across menstrual phases |

### 3-Method Voting Feature Selection

A feature was retained only if selected by **≥ 2 of 3** independent statistical tests: Pearson/Spearman correlation, Chi-square, Mutual information.

**Result:** Of the 41 raw + 7 engineered candidates, only **Follicle Asymmetry** passed among the engineered set. Final feature set = **10 raw + 1 engineered = 11 predictors**.

### Events-Per-Variable (EPV) Justification

$$
EPV_{raw} = \frac{177}{41} = 4.31 \quad (\text{below clinical minimum of } 10)
$$
$$
EPV_{engineered} = \frac{177}{11} = 16.09 \quad (\text{satisfies EPV} \geq 10)
$$

---

## 🤖 Models Evaluated (14 Classifiers, 9 Families)

| Family | Models |
|---|---|
| Linear | Logistic Regression, Perceptron |
| Kernel | Support Vector Machine (SVM) |
| Probabilistic | Gaussian Naïve Bayes, QDA |
| Discriminant | LDA |
| Tree-based | Decision Tree, Random Forest |
| Boosting | AdaBoost, XGBoost, Gradient Boosting |
| Bagging | Bagging Classifier |
| Meta-learner | Stacking ML |
| Instance-based | K-Nearest Neighbors (KNN) |

---

# 📊 PIPELINE 1: Baseline Models (Default Hyperparameters)

## 1a. Baseline — WITHOUT Feature Engineering (41 raw features)
*(Notebook 05)*

| Model | Family | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|---|
| **AdaBoost** | Boosting | 0.8957 | 0.8600 | 0.8113 | **0.8350** | 0.9487 |
| LDA | Discriminant | 0.8834 | 0.7931 | 0.8679 | 0.8288 | 0.9492 |
| XGBoost | Boosting | 0.8773 | 0.7895 | 0.8491 | 0.8182 | 0.9470 |
| Bagging Classifier | Bagging | 0.8773 | 0.7895 | 0.8491 | 0.8182 | 0.9324 |
| Stacking ML | Meta-learner | 0.8834 | 0.8400 | 0.7925 | 0.8155 | 0.9494 |
| Random Forest | Tree | 0.8650 | 0.7385 | 0.9057 | 0.8136 | 0.9483 |
| KNN | Instance-based | 0.8650 | 0.8298 | 0.7358 | 0.7800 | 0.9322 |
| Decision Tree | Tree | 0.8650 | 0.7818 | 0.8113 | 0.7963 | 0.8511 |
| Gradient Boosting | Boosting | 0.8528 | 0.7458 | 0.8302 | 0.7857 | 0.9453 |
| Logistic Regression | Linear | 0.8650 | 0.8163 | 0.7547 | 0.7843 | 0.9365 |
| Perceptron | Linear | 0.8589 | 0.7885 | 0.7736 | 0.7810 | 0.9190 |
| SVM | Kernel | 0.8650 | 0.8605 | 0.6981 | 0.7708 | 0.9352 |
| Gaussian Naïve Bayes | Probabilistic | 0.8098 | 0.6667 | 0.8302 | 0.7395 | 0.8814 |
| QDA | Probabilistic | 0.8221 | 0.7609 | 0.6604 | 0.7071 | 0.8590 |

**Mean F1 (Baseline, Without FE): 0.799**

**AdaBoost — 10-Fold Stratified CV, Confusion Matrix & ROC Curve:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/b2805255-6816-4689-95f6-e708c2350e5c" width="800" alt="AdaBoost Baseline Without FE - 10-Fold Stratified CV"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/a4ae68fd-5716-4bea-92f2-6fabece1f442" width="420" alt="AdaBoost Baseline Without FE - Confusion Matrix"/>
  <img src="https://github.com/user-attachments/assets/a21b27f0-c555-4397-9315-21f3adeecd11" width="420" alt="AdaBoost Baseline Without FE - ROC Curve"/>
</p>

**Feature Importances (AdaBoost, Without FE):**

<p align="center">
  <img src="https://github.com/user-attachments/assets/2badcdc5-38bb-4120-89af-a7b99904ccfa" width="600" alt="AdaBoost Baseline Without FE - Feature Importances"/>
</p>

---

## 1b. Baseline — WITH Feature Engineering (11 selected features)
*(Notebook 03)*

| Model | Family | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|---|
| **AdaBoost** | Boosting | 0.9080 | **0.9318** | 0.7736 | 0.8454 | 0.9475 |
| **Logistic Regression** | Linear | 0.9080 | 0.8393 | **0.8868** | **0.8624** | 0.9492 |
| Gaussian Naïve Bayes | Probabilistic | 0.9018 | 0.8776 | 0.8113 | 0.8431 | **0.9513** |
| Perceptron | Linear | 0.8957 | 0.8750 | 0.7925 | 0.8317 | 0.9288 |
| LDA | Discriminant | 0.8896 | 0.8571 | 0.7925 | 0.8235 | 0.9461 |
| KNN | Instance-based | 0.8896 | 0.9268 | 0.7170 | 0.8085 | 0.9354 |
| XGBoost | Boosting | 0.8834 | 0.8148 | 0.8302 | 0.8224 | 0.9468 |
| Stacking ML | Meta-learner | 0.8834 | 0.8400 | 0.7925 | 0.8155 | 0.9472 |
| QDA | Probabilistic | 0.8834 | 0.8148 | 0.8302 | 0.8224 | 0.9427 |
| SVM | Kernel | 0.8773 | 0.8000 | 0.8302 | 0.8148 | 0.9441 |
| Bagging Classifier | Bagging | 0.8650 | 0.7627 | 0.8491 | 0.8036 | 0.9435 |
| Gradient Boosting | Boosting | 0.8650 | 0.7627 | 0.8491 | 0.8036 | 0.9419 |
| Random Forest | Tree | 0.8589 | 0.7500 | 0.8491 | 0.7965 | 0.9477 |
| Decision Tree | Tree | 0.8466 | 0.7414 | 0.8113 | 0.7748 | 0.8375 |

**Mean F1 (Baseline, With FE): 0.817** → **+2.3% absolute improvement over 1a**

**Logistic Regression — 10-Fold Stratified CV, Confusion Matrix & ROC Curve:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/16eb1009-01ab-49c5-8592-1e954df84e34" width="800" alt="Logistic Regression Baseline With FE - 10-Fold Stratified CV"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/9559891e-164c-4062-8a48-92f6f2829281" width="420" alt="Logistic Regression Baseline With FE - Confusion Matrix"/>
  <img src="https://github.com/user-attachments/assets/4bb4d35c-b9f3-41c5-a0d3-94b19eb235b3" width="420" alt="Logistic Regression Baseline With FE - ROC Curve"/>
</p>

**Feature Importances (Logistic Regression, With FE — 11 features):**

<p align="center">
  <img src="https://github.com/user-attachments/assets/a64204a2-877a-4155-8d6d-a1e5b23853f9" width="700" alt="Logistic Regression Baseline With FE - Feature Importances"/>
</p>

### 🔑 Pipeline 1 Takeaway
- **AdaBoost** → best precision & overall F1 without FE; stays strong with FE.
- **Logistic Regression** → biggest beneficiary of FE, becomes the best **recall/screening** model (88.7%).
- Probabilistic models (Gaussian NB, QDA) were weakest without FE — confirming sensitivity to raw multicollinearity.

---

# 📊 PIPELINE 2: Tuned Models (Bayesian Optimization — Optuna TPE)

All 14 classifiers optimized using **Optuna's Tree-structured Parzen Estimator (TPE)**, 100 trials/model, optimizing stratified 10-fold CV F1-score.

## 2a. Tuned — WITHOUT Feature Engineering (41 raw features)
*(Notebook 06)*

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| AdaBoost | 0.8834 | 0.8036 | 0.8491 | 0.8257 | 0.9400 |
| LDA | 0.8834 | 0.8036 | 0.8491 | 0.8257 | 0.9561 |
| XGBoost | 0.8957 | 0.8333 | 0.8491 | 0.8411 | 0.9427 |
| Bagging | 0.8896 | 0.8182 | 0.8491 | 0.8333 | 0.9427 |
| Stacking ML | 0.9080 | 0.8958 | 0.8113 | 0.8515 | **0.9580** |
| Random Forest | 0.8896 | 0.8302 | 0.8302 | 0.8302 | 0.9396 |
| KNN | 0.7975 | 0.6351 | 0.8868 | 0.7402 | 0.9126 |
| Decision Tree | 0.8037 | 0.6780 | 0.7547 | 0.7143 | 0.8460 |
| Gradient Boosting | 0.8896 | 0.8302 | 0.8302 | 0.8302 | 0.9451 |
| **Logistic Regression** | 0.9202 | **0.9762** | 0.7736 | **0.8632** | 0.9612 |
| Perceptron | 0.8405 | 0.7547 | 0.7547 | 0.7547 | 0.9331 |
| SVM | 0.8712 | 0.7963 | 0.8113 | 0.8037 | 0.9501 |
| Gaussian Naïve Bayes | 0.8037 | 0.6364 | 0.9245 | 0.7538 | 0.8789 |
| QDA | 0.8466 | 0.7500 | 0.7925 | 0.7706 | 0.9038 |

**Logistic Regression — Confusion Matrix, 10-Fold CV & ROC Curve** (highest precision, 97.6%, via L1-induced sparsity acting like implicit feature selection):

<p align="center">
  <img src="https://github.com/user-attachments/assets/d72379dd-ec75-4762-9285-5b7da60228f3" width="420" alt="Logistic Regression Tuned Without FE - Confusion Matrix"/>
  <img src="https://github.com/user-attachments/assets/765e199e-7b1a-4550-a4ff-0ea425137d0f" width="420" alt="Logistic Regression Tuned Without FE - ROC Curve (AUC 0.9612)"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/7abc8500-7e9c-4558-9677-f8653e5dfe72" width="800" alt="Logistic Regression Tuned Without FE - 10-Fold Stratified CV"/>
</p>

**Optuna Bayesian Hyperparameter Search (Logistic Regression, Without FE):**

<p align="center">
  <img src="https://github.com/user-attachments/assets/bc4bd594-1ff2-4539-aeaf-957a5451db9d" width="380" alt="CV Score vs tol"/>
  <img src="https://github.com/user-attachments/assets/92625bb6-a132-438d-b12b-4e8e521f8f0a" width="380" alt="CV Score vs C"/>
  <img src="https://github.com/user-attachments/assets/7d34de38-c78e-4ec6-afcb-2b080c06b5d6" width="380" alt="CV Score vs penalty"/>
</p>

**Feature Importances (Logistic Regression, Tuned Without FE):**

<p align="center">
  <img src="https://github.com/user-attachments/assets/0f985fdc-a795-434a-a9a9-9358366d3380" width="700" alt="Logistic Regression Tuned Without FE - Feature Importances"/>
</p>

---

## 2b. Tuned — WITH Feature Engineering (11 selected features)
*(Notebook 04)*

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| AdaBoost | 0.8834 | 0.8400 | 0.7925 | 0.8155 | 0.9504 |
| LDA | 0.9080 | 0.8519 | 0.8679 | 0.8598 | 0.9504 |
| XGBoost | 0.9080 | 0.8800 | 0.8302 | 0.8544 | 0.9544 |
| Bagging | 0.8896 | 0.8302 | 0.8302 | 0.8302 | 0.9499 |
| Stacking ML | 0.9018 | **0.9111** | 0.7736 | 0.8367 | 0.9561 |
| Random Forest | 0.8712 | 0.8077 | 0.7925 | 0.8000 | 0.9463 |
| KNN | 0.8589 | 0.8261 | 0.7170 | 0.7677 | 0.9357 |
| Decision Tree | 0.8650 | 0.7818 | 0.8113 | 0.7963 | 0.9101 |
| Gradient Boosting | 0.9018 | 0.8776 | 0.8113 | 0.8431 | 0.9528 |
| Logistic Regression | 0.9202 | 0.9167 | 0.8302 | 0.8713 | 0.9540 |
| Perceptron | 0.9202 | 0.9000 | 0.8491 | 0.8738 | 0.9532 |
| **SVM (RBF)** | **0.9202** | 0.8571 | **0.9057** | **0.8807** | 0.9492 |
| Gaussian Naïve Bayes | 0.8773 | 0.9231 | 0.6792 | 0.7826 | 0.9516 |
| QDA | 0.9018 | 0.8491 | 0.8491 | 0.8491 | 0.9475 |

**SVM (RBF) — Best overall model — Confusion Matrix, 10-Fold CV & ROC Curve:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/501e9a69-90c2-428d-b242-ae46a499aeb8" width="420" alt="SVM Tuned With FE - Confusion Matrix"/>
  <img src="https://github.com/user-attachments/assets/efb0e149-753c-4718-a600-2d110fe95291" width="420" alt="SVM Tuned With FE - ROC Curve"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/06f97001-e0ff-463e-b6b7-339b8b7483f6" width="800" alt="SVM Tuned With FE - 10-Fold Stratified CV"/>
</p>

**Optuna Bayesian Hyperparameter Search (SVM, With FE):**

<p align="center">
  <img src="https://github.com/user-attachments/assets/ccd08fd5-bf70-45dc-a7bc-e292ac0f660e" width="420" alt="CV Score vs kernel"/>
  <img src="https://github.com/user-attachments/assets/d6789e98-d445-40c9-b6b6-1ccbef657e7e" width="420" alt="CV Score vs C"/>
</p>

**Feature Importances (SVM, Tuned With FE):**

<p align="center">
  <img src="https://github.com/user-attachments/assets/2cdb6f74-387e-465b-80eb-6b76848dc01c" width="600" alt="SVM Tuned With FE - Feature Importances"/>
</p>

### 🔑 Pipeline 2 Takeaway
- **Without FE:** Logistic Regression dominates (F1 = 0.8632) — L1 penalty (C=0.111) internally mimics feature selection on the 41-feature space.
- **With FE:** SVM (RBF kernel) becomes best overall (F1 = 0.8807) — Bayesian tuning switches its kernel from linear → RBF once dimensionality drops to 11, capturing non-linear hormonal interactions.
- **Mean F1: 0.808 (without FE) → 0.833 (with FE)**, a **+0.031** gain. Individual gains vary from +0.1191 (Perceptron) to −0.0302 (Random Forest).

---

## 📈 4-Condition Comparison Summary

| Pipeline | Condition | Mean F1 (14 models) | Best Model | Best F1 | Best Recall |
|---|---|---|---|---|---|
| 1. Baseline | Without FE | 0.799 | AdaBoost | 0.835 | Random Forest (0.906) |
| 1. Baseline | With FE | 0.817 | AdaBoost | 0.845 | Logistic Regression (0.887) |
| 2. Tuned | Without FE | 0.808 | Logistic Regression | 0.863 | Gaussian NB (0.925) |
| 2. Tuned | With FE | 0.833 | **SVM (RBF)** | **0.881** | **SVM (0.906)** |

```mermaid
xychart-beta
    title "Mean F1-Score Across All 4 Conditions"
    x-axis ["Baseline-NoFE", "Baseline-FE", "Tuned-NoFE", "Tuned-FE"]
    y-axis "Mean F1 Score" 0.75 --> 0.85
    bar [0.799, 0.817, 0.808, 0.833]
```

**ROC-AUC across all 4 experiments, per model:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ff6c0cc-84bd-45a5-9091-968e98badd96" width="900" alt="ROC-AUC - All 4 Experiments per Model"/>
</p>

**Combined ROC curves — all 14 models, baseline configurations:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/db1c1db0-38e7-4047-830e-ee39f80772d7" width="650" alt="ROC Curves - All Baseline Model Configurations"/>
</p>

---

## ⏱️ Training Time & Optimization Cost (Notebook 08)

Reducing the feature space from 41 → 11 substantially cut computational cost during Bayesian hyperparameter search:

| Model | Optimization Time (Without FE, 41 features) | Optimization Time (With FE, 11 features) | Reduction |
|---|---|---|---|
| AdaBoost | 3938 sec | 2169 sec | **−45%** |
| Random Forest | High | Reduced | **−66%** |
| Bagging Classifier | High | Reduced | **−27%** |
| SVM | Low (best F1/time tradeoff) | Fastest, highest F1 (0.8807) | Best overall efficiency |

**Key finding:** Feature engineering doesn't just improve accuracy — it makes the **entire Bayesian AutoML search meaningfully cheaper**, since matrix operations (kernel calculations in SVM, covariance estimation in LDA/QDA, split-finding in trees) scale with feature count.

**Model performance vs. training data size, across all 14 models (both feature sets):**

<p align="center">
  <img src="https://github.com/user-attachments/assets/fb013432-8751-4c6d-913a-76dc162053ce" width="850" alt="Without FE - All Models F1 Score vs Training Data Size"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/645433ee-fe22-4b3e-8290-ff15ea0220ad" width="850" alt="With FE - All Models ROC-AUC vs Training Data Size"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/d6053ffe-f74d-4a6b-994e-982882d0ad1c" width="850" alt="With FE - All Models F1 Score vs Training Data Size"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/cbfc7184-6552-4534-a22a-57e9af9f2104" width="850" alt="Without FE - All Models ROC-AUC vs Training Data Size"/>
</p>

---

## 📉 Learning Curve Analysis (Notebook 10)

Learning curves were plotted for all 14 classifiers to evaluate overfitting and generalization as training data size increased (10% → 100%).

| Observation | Detail |
|---|---|
| **Convergence** | Most classifiers converged to a stable ~0.95 ROC-AUC beyond 50–60% of training data, with minimal overfitting |
| **Decision Tree overfitting** | Training performance stuck at 1.00 F1/AUC, while test performance plateaued at only 0.76–0.86 |
| **AdaBoost: FE vs No-FE stability** | Non-FE AdaBoost was *more stable* at full training data (F1 = 0.8257) vs. the FE-tuned version (F1 = 0.8155) — for lower training fractions (10–30%) the gap shrank, suggesting FE mainly benefits **consistency/stability**, not asymptotic performance |
| **Data sufficiency** | None of the 14 classifiers had clearly plateaued — an estimated **~248 additional patients (45.8% more data)** would likely be needed to reach a true performance plateau |

**SVM — Learning curves (Without FE, and With vs. Without FE comparison):**

<p align="center">
  <img src="https://github.com/user-attachments/assets/1e9e7d99-0f73-4af7-beb4-04ac516a59a0" width="850" alt="SVM Learning Curve - Without FE"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/655db017-3ecb-4a82-be00-141c6d6a2786" width="850" alt="SVM Learning Curve Comparison - With vs Without FE"/>
</p>

**Logistic Regression — Learning curves (With vs Without FE comparison, and Without FE standalone):**

<p align="center">
  <img src="https://github.com/user-attachments/assets/8e7e3251-f279-419a-9553-09358eb3adf4" width="850" alt="Logistic Regression Learning Curve Comparison - With vs Without FE"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/f537ccae-12ab-4584-8701-26eef05737ac" width="850" alt="Logistic Regression Learning Curve - Without FE"/>
</p>

**Overfitting analysis (Decision Tree — train vs. test divergence):**

<p align="center">
  <img src="https://github.com/user-attachments/assets/73835454-6e39-45a7-81a0-9d78ddf9263d" width="750" alt="Overfitting Analysis - Decision Tree Train vs Test"/>
</p>

**Stratified 10-fold cross-validation strategy overview:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/6c0bdf7b-cfa7-42f1-9e47-024c6bbba98c" width="750" alt="Stratified 10-Fold Cross-Validation Overview"/>
</p>

---

## 🧩 The Four Archetype Framework

Derived by comparing model behavior across all 4 conditions (Baseline/Tuned × With/Without FE):

| Archetype | Models | Description |
|---|---|---|
| **1. Bimodal Sensitivity** | SVM, Perceptron, LDA, QDA, Decision Tree | Strongly benefit from **both** FE and tuning |
| **2. Structural Rigidity** | Gaussian Naïve Bayes, KNN | Modest FE gains; Bayesian tuning often *underperforms* defaults |
| **3. Algorithmic Robustness** | XGBoost, Gradient Boosting | Built-in regularization makes them largely insensitive to FE |
| **4. Default Optimality** | AdaBoost, Random Forest, Bagging, Stacking ML | Internal per-node/per-estimator feature selection makes external FE largely redundant |

```mermaid
quadrantChart
    title Four Archetype Framework
    x-axis Low FE Sensitivity --> High FE Sensitivity
    y-axis Low Tuning Sensitivity --> High Tuning Sensitivity
    quadrant-1 Archetype 1 Bimodal Sensitivity
    quadrant-2 Archetype 3 Algorithmic Robustness
    quadrant-3 Archetype 2 Structural Rigidity
    quadrant-4 Archetype 4 Default Optimality
    SVM: [0.85, 0.85]
    Perceptron: [0.9, 0.6]
    LDA: [0.6, 0.5]
    QDA: [0.75, 0.4]
    Decision Tree: [0.6, 0.3]
    Gaussian NB: [0.2, 0.15]
    KNN: [0.2, 0.2]
    XGBoost: [0.15, 0.7]
    Gradient Boosting: [0.15, 0.65]
    AdaBoost: [0.1, 0.2]
    Random Forest: [0.1, 0.15]
    Bagging: [0.15, 0.25]
    Stacking ML: [0.15, 0.25]
    Logistic Regression: [0.3, 0.9]
```

---

## 🔍 Explainability & Diagnostics

Applied to the best-performing tuned models to move beyond black-box classification:

| Tool | Purpose |
|---|---|
| **SHAP / LIME** | Global & local feature importance — top predictors: Follicle No. (R/L), Hair growth, Weight gain, Skin darkening, AMH, Follicle Asymmetry |
| **Calibration Curves** | Logistic Regression best calibrated (Brier Score = 0.069) |
| **Decision Curve Analysis (DCA)** | Clinical net benefit across probability thresholds |

<p align="center">
  <img src="https://github.com/user-attachments/assets/050ed09d-bc81-47eb-a980-f0003e0cf671" width="900" alt="Probability Calibration Curves - Risk Score Reliability"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/c27880cd-8051-42dc-9a5f-1f36ddd32f47" width="900" alt="Decision Curve Analysis - Clinical Net Diagnostic Utility"/>
</p>
<p align="center">
  <img width="1179" height="944" alt="Image" src="https://github.com/user-attachments/assets/467359d1-8c04-407d-abcf-8114c7bada4f" />
</p>
<p align="center">
 <img width="1028" height="694" alt="Image" src="https://github.com/user-attachments/assets/92929d07-0bcd-42d6-b3c5-ad928c03efc2" />
</p>
<p align="center">
   <img width="1140" height="796" alt="Image" src="https://github.com/user-attachments/assets/8e940284-b3bc-4bc9-8418-35a971603315" />
</p>

---

## 📐 Evaluation Metrics

| Metric | Definition |
|---|---|
| Accuracy | Proportion of correctly classified instances |
| Precision | TP / (TP + FP) |
| Recall (Sensitivity) | TP / (TP + FN) — clinically prioritized, since false negatives are costly |
| F1-Score | Harmonic mean of Precision & Recall |
| AUC-ROC | Threshold-independent discriminative measure |
| Brier Score | Calibration quality of predicted probabilities |

---

## 🚀 Getting Started

```bash
git clone https://github.com/hariniisathappan/PCOS_Classification_Using_ML.git
cd PCOS_Classification_Using_ML
pip install -r requirements.txt
```

Run notebooks in this order for full reproducibility:

```bash
01_data_exploration.ipynb
02_preprocessing_pipeline.ipynb
07_FE_and_FS.ipynb
03_with_FE_Baseline.ipynb
04_with_FE_Tuned.ipynb
05_without_FE_baseline.ipynb
06_without_FE_Tuned.ipynb
08_Time_and_performance.ipynb
09_Trainind_data_metrics.ipynb
10_Learning_curve.ipynb
```

---

## 🧠 Conclusion

- Rotterdam-criteria-based **Feature Engineering** improves mean F1 across both baseline (+2.3%) and tuned (+3.1%) pipelines, while raising the **EPV ratio from 4.31 to 16.09**.
- **AdaBoost** is the strongest baseline (default) model; **SVM (RBF)** is the strongest model once Bayesian-tuned with engineered features.
- **Logistic Regression** is remarkably consistent across all four conditions and offers the best-calibrated probability outputs (Brier = 0.069).
- Feature engineering also reduces **training/optimization time by up to 66%** for some models.
- Learning curves show that **most models have not yet plateaued** — more clinical data (~46% more patients) would likely improve robustness further.

---

## 👩‍💻 Author

**Harini S** — Division of Bioinformatics, SASTRA Deemed University, Thanjavur

---

## 📄 License

This project is licensed under the MIT License.
