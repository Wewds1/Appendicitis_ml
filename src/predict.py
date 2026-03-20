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