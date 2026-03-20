import json
import joblib
import pandas as pd

# This function takes a trained machine learning model and a DataFrame of features, and returns the predicted labels for those features.
def predict_labels(model, X: pd.DataFrame):
    return model.predict(X)


# This function takes a trained machine learning model and a DataFrame of features, and returns the predicted probabilities for the positive class.
def predict_proba(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    else:
        raise ValueError("Model does not have predict_proba method.")

# This function takes a trained machine learning model, a DataFrame of features, and a threshold value, and returns the predicted labels based on whether the predicted probabilities exceed the threshold.
def predict_with_threshold(model, X: pd.DataFrame, threshold: float = 0.5):
    scores = predict_proba(model, X)
    return (scores >= threshold).astype(int)


# This function loads a trained machine learning model from the specified file path using joblib.
def load_model(model_path: str):
    return joblib.load(model_path)


# This function loads the list of feature columns used during model training from a JSON file at the specified path.
def load_feature_columns(features_path: str):
    with open(features_path, 'r', encoding='utf-8') as f:
        feature_columns = json.load(f)
    return feature_columns['columns']


# This function takes an input DataFrame and a list of training feature columns, and returns a new 
# DataFrame that is aligned with the training features.
def align_features(input_df: pd.DataFrame, training_columns: list) -> pd.DataFrame:
    aligned = input_df.copy()
    for c in training_columns:
        if c not in aligned.columns:
            aligned[c] = 0
            
    return aligned[training_columns]
