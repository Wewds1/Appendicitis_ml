# Appendicitis Risk Classification Model Card

## 1. Model Details
- **Project:** Appendicitis ML analysis pipeline
- **Problem Type:** Binary classification
- **Target:** Final_Diagnosis (mapped to `target_bin`)
- **Positive Class:** Appendicitis (1)
- **Candidate Models:** LogisticRegression, RandomForest, GradientBoosting
- **Selection Rule:** Priority order: High CV Recall -> PR-AUC -> F1

## 2. Intended Use
- Educational and portfolio demonstration of end-to-end ML workflows.
- Triage-style risk estimation from available clinical features.
- Support for analysis and learning; not for autonomous medical decision-making.

## 3. Out-of-Scope Use
- Real clinical diagnosis without licensed physician oversight.
- Emergency triage deployment without prospective validation.
- Direct patient-care automation or standalone diagnostic output.

## 4. Data Summary
- **Source:** appendicitis_comprehensive_dataset.csv
- **Samples:** 1,500 rows | **Raw Features:** 23
- **Class Distribution:**
  - Appendicitis: 1,190 (79.33%)
  - Other: 310 (20.67%)
- **Split:** 80/20 Stratified (random_state=42)

## 5. Preprocessing and Leakage Controls
- **Dropped Columns (Leakage):** Patient_ID, Management, Pathological_Cause, Severity
- **Encoding:** Yes/No fields mapped to 1/0
- **Numeric Pipeline:** Median imputation + Standard scaling
- **Categorical Pipeline:** Most-frequent imputation + One-hot encoding
- **Artifacts:** X_train.csv, X_test.csv, y_train.csv, y_test.csv

## 6. Feature Engineering
- **Engineered Features:** Age_Group, BMI_Group, Symptoms_Positive_Count, Inflammation_Score, Duration_Bucket, Young_Patient, High_CRP, High_WBC.
- **Outcome:** Performance remained similar to baseline; additional complexity did not yield significant gains.

## 7. Evaluation Protocol
- **Primary Metric:** Recall
- **Secondary Metrics:** PR-AUC, F1, Precision, ROC-AUC, Accuracy
- **Validation:** 5-fold Stratified Cross-Validation + Holdout Test comparison

## 8. Performance Snapshot

### Cross-Validation (Mean)
Model               Accuracy  Precision  Recall  F1      ROC-AUC  PR-AUC
RandomForest        0.9950    0.9938     1.0000  0.9969  0.9998   0.9999
GradientBoosting    0.9942    0.9938     0.9989  0.9964  0.9992   0.9998
LogisticRegression  0.9958    0.9979     0.9968  0.9974  0.9996   0.9999

### Holdout Test (Final Comparison)
Model               Accuracy  Precision  Recall  F1      ROC-AUC  PR-AUC
LogisticRegression  0.9967    1.0000     0.9958  0.9979  1.0000   1.0000
RandomForest        0.9967    0.9958     1.0000  0.9979  1.0000   1.0000
GradientBoosting    1.0000    1.0000     1.0000  1.0000  1.0000   1.0000

### Confusion Matrix (LogisticRegression)
- TN: 62 | FP: 0
- FN: 1  | TP: 237

## 9. Known Limitations
- High scores suggest highly separable or synthetic-like data; real-world hospital data may show significant performance drop.
- Class imbalance requires careful thresholding despite high metrics.
- Lacks external validation on independent datasets.

## 10. Ethical and Safety Notes
- **Disclaimer:** Not a medical device. Must not replace clinician judgment.
- **Mandatory Requirements:** Human oversight, external validation, calibration analysis, and subgroup performance audits are required before any practical application.

## 11. Reproducibility
- **Modules:** data_prep.py, features.py, train.py, predict.py
- **Tests:** test_data_prep.py

## 12. Next Improvements
- Integrate calibration metrics and reliability plots.
- Perform external validation on independent datasets.
- Define threshold policies based on clinical costs of False Negatives vs. False Positives.
- Implement pipeline serialization (Joblib/Pickle) and inference examples.