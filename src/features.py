import pandas as pd 


# Define columns that are considered data leakage and should be dropped from the dataset before modeling.
# Copy paste this list from data_prep.py to here, so we can use it in the feature engineering notebook as well.
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Age_Group"] = pd.cut(
        out["Age"],
        bins=[0, 12, 18, 35, 60, 120],
        labels=["Child", "Teen", "Young_Adult", "Adult", "Senior"],
        include_lowest=True,
    )

    out["BMI_Group"] = pd.cut(
        out["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
        include_lowest=True,
    )

    signs = [
        "Pain_Migration",
        "Nausea_Vomiting",
        "Loss_of_Appetite",
        "Rebound_Tenderness",
        "McBurney_Sign",
        "Rovsing_Sign",
        "Psoas_Sign",
    ]
    out["Symptoms_Positive_Count"] = out[signs].sum(axis=1)

    out["Inflammation_Score"] = (
        out["WBC_Count_k_uL"] + out["Neutrophil_Percentage"] / 10 + out["CRP_Level_mg_L"] / 10
    )

    out["Duration_Bucket"] = pd.cut(
        out["Duration_of_Symptoms_Hours"],
        bins=[0, 6, 12, 24, 48, 1000],
        labels=["Very_Recent", "Recent", "12_24h", "Moderate", "Prolonged"],
        include_lowest=True,
    )

    out["Young_Patient"] = (out["Age"] <= 18).astype(int)
    out["High_CRP"] = (out["CRP_Level_mg_L"] >= out["CRP_Level_mg_L"].median()).astype(int)
    out["High_WBC"] = (out["WBC_Count_k_uL"] >= out["WBC_Count_k_uL"].median()).astype(int)

    return out


# This function takes a DataFrame and prepares it for modeling by separating the features and target variable.
# It drops the original target column and the binary target column from the features, and then applies one-hot encoding to the remaining features.
# The function returns the encoded feature matrix X and the target vector y.
# Just copy and paaste what i have on modeling.ipynb to here, and then we can use it in the notebook.
def to_model_matrix(df: pd.DataFrame, target_col: str = 'Final_Diagnosis', target_bin_col : str = 'target_bin'):
    X = df.drop(columns=[target_col, target_bin_col], errors="ignore")
    y = df[target_bin_col]
    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded, y