import pandas as pd
import numpy as np

from typing import Dict, List, Tuple
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate, StratifiedKFold


# This function returns a dictionary of machine learning models that can be used for training and evaluation.
def get_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(random_state=random_state, max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(random_state=random_state, n_estimators=300, class_weight="balanced", n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
    }
    
    

# This function performs cross-validation on a set of machine learning models and returns a DataFrame with the average performance metrics for each model.
def cross_validate_models(models: Dict[str, object], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision'
    }
    
    rows = []
    for name, model in models.items():
        out = cross_validate(model,X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({
            "model": name,
            "cv_accuracy_mean": out["test_accuracy"].mean(),
            "cv_precision_mean": out["test_precision"].mean(),
            "cv_recall_mean": out["test_recall"].mean(),
            "cv_f1_mean": out["test_f1"].mean(),
            "cv_roc_auc_mean": out["test_roc_auc"].mean(),
            "cv_pr_auc_mean": out["test_pr_auc"].mean()
        })
        
    return pd.DataFrame(rows).sort_values(by=["cv_recall_mean", "cv_precision_mean", "cv_f1_mean", "cv_pr_auc_mean"], ascending=False)

# This function fits a given machine learning model on the training data, makes predictions on the test data, 
# and evaluates the performance using various metrics. It returns the fitted model and a dictionary of performance metrics.
def fit_and_eval_model(model, X_train, y_train, X_test, y_test) -> Tuple[object, dict]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
        "pr_auc": average_precision_score(y_test, y_proba) if y_proba is not None else np.nan
    }
    
    
    return model, metrics
