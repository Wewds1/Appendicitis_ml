from pathlib import Path
from typing import Dict, List, Tuple


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LEAKAGE_COLS = ["Patient_ID", "Management", "Pathological_Cause", "Severity"]
BOOLEAN_COLS = [
    "Is_Pregnant",
    "Pain_Migration",
    "Nausea_Vomiting",
    "Loss_of_Appetite",
    "Rebound_Tenderness",
    "McBurney_Sign",
    "Rovsing_Sign",
    "Psoas_Sign",
]

# This function loads the raw data from a specified CSV file path and returns it as a pandas DataFrame.
def load_raw_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

# This function drops columns that are considered data leakage, meaning they contain 
# information that would not be available at the time of prediction. By default, 
# it drops the columns listed in LEAKAGE_COLS, but you can specify a custom list of columns if needed.
def drop_leakage_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=LEAKAGE_COLS, errors="ignore")

# This function encodes specified boolean columns by mapping "yes" to 1 and "no" to 0. 
# It also handles any leading/trailing whitespace and case variations in the string values. 
# By default, it will encode all columns listed in BOOLEAN_COLS, but you can specify a custom list of columns if needed.
def encode_boolean_cols(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    cols = columns or BOOLEAN_COLS
    for col in cols:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    return out

# This function adds a binary target column based on the specified positive class in the original target column.
# By default, it creates a "target_bin" column where 1 indicates "Appendicitis" and 0 indicates any other diagnosis.
def add_binary_target(df: pd.DataFrame, target_col: str = "Final_Diagnosis",
                    positive_class: str = "Appendicitis",
                    target_bin_col: str ="target_bin") -> pd.DataFrame:
    out = df.copy()
    
    out[target_bin_col] = (out[target_col] == positive_class).astype(int)
    return out


# This is the main function to call for data cleaning. It applies all the necessary transformations in sequence.
def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = drop_leakage_cols(df_raw)
    df = encode_boolean_cols(df)
    df = add_binary_target(df)
    return df

# Note: This function is not currently used, but could be helpful for future data splits or 
# if we want to return a cleaned DataFrame with the target column included.
def split_X_y(df: pd.DataFrame,
            target_col: str = "Final_Diagnosis",
            target_bin_col: str = "target_bin") -> Tuple[pd.DataFrame, pd.Series]:
    
    X = df.drop(columns=[target_col, target_bin_col], errors="ignore")
    y = df[target_bin_col]
    
    return X, y


# This function performs a stratified train-test split, ensuring that the proportion of classes in the target 
# variable is maintained in both the training and testing sets. By default, it uses a test size of 20% 
# and a random state of 42 for reproducibility like a normal human.
def train_test_split_stratified(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# This function builds a preprocessing pipeline that handles both numeric and categorical features. 
# Numeric features are imputed using the median and scaled using StandardScaler, 
# while categorical features are imputed using the most frequent value and encoded using OneHotEncoder. 
# The function returns a ColumnTransformer that can be applied to the training and testing data.
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    
    
    numeric_pipeline = Pipeline (
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    
    preprocessor = ColumnTransformer (
        transformers = [
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )
    
    
    return preprocessor

# This function saves the cleaned DataFrame and the train-test splits to specified directories. 
# It creates the directories if they do not exist and returns a dictionary with the file paths of the saved CSV files.
def save_clean_data(df_clean: pd.DataFrame,
                    X_train_df: pd.DataFrame,
                    X_test_df: pd.DataFrame,
                    y_train: pd.Series,
                    y_test: pd.Series,
                    interim_dir: str = "../data/interim/",
                    processed_dir: str = "../data/processed/") -> Dict[str,str]:
    Path(interim_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    
    path1 = Path(interim_dir) / "appendicitis_clean.csv"
    path2 = Path(processed_dir) / "X_train.csv"
    path3 = Path(processed_dir) / "X_test.csv"
    path4 = Path(processed_dir) / "y_train.csv"
    path5 = Path(processed_dir) / "y_test.csv"
    
    df_clean.to_csv(path1, index=False)
    X_train_df.to_csv(path2, index=False)
    X_test_df.to_csv(path3, index=False)
    y_train.to_csv(path4, index=False)
    y_test.to_csv(path5, index=False)
    
    return {
        "clean_data": str(path1),
        "X_train": str(path2),
        "X_test": str(path3),
        "y_train": str(path4),
        "y_test": str(path5)
    } 