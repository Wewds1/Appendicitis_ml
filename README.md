# Appendicitis ML Pipeline

## Project Goal
This project builds an end-to-end machine learning workflow to classify appendicitis risk from patient symptoms, exam findings, lab values, and ultrasound-related features.

The goal is to show both:
* Data science thinking (EDA, feature engineering, model comparison)
* ML engineering discipline (modular src code, tests, reproducibility, documentation)

## Problem Definition
* Task: Binary classification
* Target: Final_Diagnosis converted to target_bin
* Positive class: Appendicitis (1)
* Negative class: Other diagnoses (0)

## Dataset Summary
* Source: appendicitis_comprehensive_dataset.csv
* Rows: 1500
* Columns: 24 in raw dataset
* Class distribution:
  - Appendicitis: 1190 (79.33%)
  - Other diagnoses combined: 310 (20.67%)

## Repository Structure
├── data/
│   ├── raw/               # original raw dataset
│   ├── interim/           # cleaned and feature-engineered outputs
│   └── processed/         # model-ready train/test matrices
├── notebooks/
│   ├── eda.ipynb
│   ├── data_cleaning.ipynb
│   ├── feature_engineering.ipynb
│   └── modeling.ipynb
├── src/
│   ├── data_prep.py       # cleaning and preprocessing utilities
│   ├── features.py        # feature engineering functions
│   ├── train.py           # model training and CV helpers
│   └── predict.py         # inference helpers
├── tests/
│   └── test_data_prep.py  # data prep tests
├── model_card.md          # model documentation
└── baseline_comparison.csv # saved baseline metrics

## Quickstart
1. Create and activate virtual environment:
   python -m venv venv
   venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Run tests:
   python -m pytest -q

4. Run notebooks in this order:
   eda.ipynb -> data_cleaning.ipynb -> feature_engineering.ipynb -> modeling.ipynb

## Methodology
1) EDA
- Missingness, duplicates, and category audits
- Class imbalance check
- Distribution, outlier, and correlation analysis
- Leakage candidate identification

2) Data Cleaning and Preparation
- Leakage columns removed: Patient_ID, Management, Pathological_Cause, Severity
- Yes/No fields mapped to 1/0
- Stratified train/test split (80/20, random_state 42)
- Numeric pipeline: median imputation + scaling
- Categorical pipeline: most-frequent imputation + one-hot encoding

3) Modeling
- Models: LogisticRegression, RandomForest, GradientBoosting
- Selection priority: Recall (1st), PR-AUC (2nd), F1 (3rd)
- Evaluation: Cross-validation on training data + Holdout test set

4) Feature Engineering
- Tested: Age_Group, BMI_Group, Inflammation_Score, High_CRP, High_WBC, etc.
- Result: Similar to baseline; no strong improvement over simpler representation.

## Results Snapshot
Cross-validation mean:
Model              Accuracy  Precision  Recall  F1      ROC-AUC  PR-AUC
RandomForest       0.9950    0.9938     1.0000  0.9969  0.9998   0.9999
GradientBoosting   0.9942    0.9938     0.9989  0.9964  0.9992   0.9998
LogisticRegression 0.9958    0.9979     0.9968  0.9974  0.9996   0.9999

Holdout test (LogisticRegression):
- Accuracy: 0.9967
- Precision: 1.0000
- Recall: 0.9958
- TN: 62 | FP: 0 | FN: 1 | TP: 237

## Limitations
- Near-perfect performance suggests highly separable or synthetic-like patterns.
- Real hospital distribution shift may reduce performance.
- Class imbalance still requires careful threshold policy.
- External validation is not yet included.

## Safety and Ethics
- Educational project only; not a medical device.
- Must not replace clinician judgment.
- Human oversight required for any practical use.

## Next Improvements
- Add external validation dataset and report generalization gap.
- Add calibration analysis and reliability curves.
- Add subgroup performance analysis by age/sex.
- Add final training script and model artifact versioning.
- Add threshold policy based on false negative clinical risk.