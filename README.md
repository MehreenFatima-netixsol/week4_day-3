# Week 4 · Day 3 — Feature Engineering, Cross-Validation & Model Comparison

## Project Overview

This project is part of the **Machine Learning Foundations** series and focuses on improving the predictive performance of the **Adult Census Income** dataset through feature engineering, robust model evaluation, and feature selection.

The objective is to engineer informative features, compare multiple machine learning models using **5-fold Stratified Cross-Validation**, perform statistical significance testing, and identify the best-performing model and feature set for subsequent hyperparameter tuning.

The project builds upon the train/test split created in **Week 4 · Day 1**, ensuring that the test dataset remains completely unseen throughout experimentation.

---

## Dataset

**Dataset:** Adult Census Income Dataset

**Prediction Task:** Binary Classification

* **Target Variable:** `income`

  * `<=50K`
  * `>50K`

### Data Split

| Dataset      | Records |
| ------------ | ------- |
| Training Set | 32,561  |
| Test Set     | 16,281  |

The training dataset was used for all feature engineering, preprocessing, cross-validation, model comparison, and feature selection. The test dataset was intentionally kept sealed to avoid information leakage and will be used only after hyperparameter tuning.

---

# Project Objectives

* Engineer meaningful features from existing variables.
* Evaluate the predictive signal of engineered features.
* Build a leakage-free preprocessing pipeline.
* Compare multiple classification algorithms using Stratified K-Fold Cross-Validation.
* Perform statistical testing between top-performing models.
* Analyze feature importance.
* Evaluate feature selection techniques.
* Recommend the optimal model and feature set for hyperparameter tuning.

---

# Engineered Features

Eight new features were created using only row-level information.

| Feature            | Description                                                       |
| ------------------ | ----------------------------------------------------------------- |
| `net_capital`      | Difference between capital gain and capital loss                  |
| `is_married`       | Indicates whether marital status begins with "Married"            |
| `edu_x_hours`      | Interaction between education level and working hours             |
| `log_capital_gain` | Log transformation of capital gain                                |
| `age_bucket`       | Age grouped into life-stage categories                            |
| `higher_education` | Indicates Bachelor's degree or above                              |
| `hours_bin`        | Categorizes working hours into part-time, full-time, and overtime |
| `has_capital_gain` | Indicates whether capital gain is greater than zero               |

All engineered features are leakage-free and generated inside the preprocessing pipeline.

---

# Machine Learning Pipeline

The preprocessing workflow consists of:

* FunctionTransformer for feature engineering
* Median Imputation for numerical features
* Most Frequent Imputation for categorical features
* StandardScaler for numerical variables
* OneHotEncoder for categorical variables
* ColumnTransformer for combining preprocessing steps
* Scikit-Learn Pipeline for end-to-end reproducibility

This design ensures that every preprocessing step is fitted only on the training folds during cross-validation.

---

# Models Evaluated

Three supervised learning algorithms were compared:

* Logistic Regression
* Random Forest Classifier
* HistGradientBoosting Classifier

Evaluation was performed using **5-Fold Stratified Cross-Validation**.

---

# Evaluation Metrics

The following metrics were used:

* Accuracy
* F1 Score (Primary Metric)
* ROC AUC
* Training Time

F1 Score was selected as the primary metric because the Adult Income dataset contains class imbalance.

---

# Statistical Analysis

The project includes:

* Paired t-test
* Wilcoxon Signed-Rank Test

These tests determine whether the performance difference between the top-performing models is statistically significant rather than due to random variation.

---

# Feature Selection

Feature selection was performed using:

* SelectKBest
* Mutual Information

Different values of **k** were evaluated to compare predictive performance and computational cost against the complete feature set.

---

# Results Summary

* HistGradientBoosting achieved the highest Accuracy, F1 Score, and ROC AUC.
* Feature engineering significantly improved model performance.
* Capital-income and interaction-based features contributed the strongest predictive signal.
* Feature selection did not improve performance and increased training time.
* The complete feature set was retained for future model tuning.

---

# Technologies Used

* Python 3
* Pandas
* NumPy
* Scikit-Learn
* SciPy
* Matplotlib
* Seaborn

---

# Key Learning Outcomes

* Applied practical feature engineering techniques.
* Built reusable machine learning pipelines.
* Prevented data leakage during preprocessing.
* Evaluated models using Stratified Cross-Validation.
* Compared models using multiple evaluation metrics.
* Performed statistical significance testing.
* Interpreted feature importance.
* Assessed feature selection methods.
* Selected the best model for future hyperparameter optimization.

---

# Future Work

The next phase of the project will focus on:

* Hyperparameter tuning using Grid Search or Random Search
* Final evaluation on the held-out test dataset
* Model calibration
* Threshold optimization
* Final performance reporting

---

# Author

**Mehreen Fatima**
