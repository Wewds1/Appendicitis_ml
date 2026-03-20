import json
from datetime import datetime
from pathlib import Path


import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix



from src.data_prep import (
    load_raw_data,
    clean_data,
    split_X_y,
    train_test_split_stratified,
    build_preprocessor,
    transform_to_dataframes
)


from src.features import engineer_features, to_model_matrix
from src.train import get_models, cross_validate_models, fit_and_eval_model


def main():
    raw_path = "data/raw/appendicitis_comprehensive_dataset.csv"
    models_dir = Path("models")
    reports_dir = Path("reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # load and clean
    df_raw = load_raw_data(raw_path)
    df_clean = clean_data(df_raw)
    
    # feature engineering
    df_feat = engineer_features(df_clean)
    
    # build model matrix
    X, y = to_model_matrix(df_feat)
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2, random_state=42)
    
    # model selection by cross-validation
    models = get_models(random_state=42)
    
    cv_df = cross_validate_models(models, X_train, y_train)
    best_model_name = cv_df.iloc[0]["model"]
    best_model = models[best_model_name]
    
    # fit and test eval
    fitted_model, test_metrics = fit_and_eval_model(best_model, X_train, y_train, X_test, y_test)
    y_pred = fitted_model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # save model and report
    model_path = models_dir / f"best_model_{run_id}.joblib"
    features_path = models_dir / f"feature_columns_{run_id}.json"
    cv_path = reports_dir / f"cv_results_{run_id}.csv"
    test_path = reports_dir / f"test_metrics_{run_id}.csv"
    
    joblib.dump(fitted_model, model_path)
    with open(features_path, 'w', encoding='utf-8') as f:
        json.dump({"columns": X_train.columns.tolist()}, f, ensure_ascii=False, indent=2)
        
        
    cv_df.to_csv(cv_path, index=False)
    
    payload = {
        "run_id": run_id,
        "best_model": best_model_name,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    
    print("Training complete")
    print("Best model:", best_model_name)
    print("Model saved to:", model_path)
    print("CV results:", cv_path)
    print("Test metrics:", test_path)
        
if __name__ == "__main__":
    main()