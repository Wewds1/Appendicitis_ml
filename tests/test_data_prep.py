import pandas as pd

from src.data_prep import (
    LEAKAGE_COLS,
    BOOLEAN_COLS,
    drop_leakage_cols,
    encode_boolean_cols,
    clean_data,
    split_X_y,
    train_test_split_stratified,
    build_preprocessor,
)


def test_leakage_columns_are_dropped():
    df = pd.DataFrame({
        "Patient_ID": ["A1", "A2"],
        "Management": ["Op", "Obs"],
        "Pathological_Cause": ["X", "Y"],
        "Severity": ["High", "Low"],
        "Age": [20, 30],
        "Final_Diagnosis": ["Appendicitis", "Other"],
    })

    out = drop_leakage_cols(df)
    for c in LEAKAGE_COLS:
        assert c not in out.columns


def test_yes_no_encoding_correctness():
    df = pd.DataFrame({
        "Is_Pregnant": ["Yes", "No", " yes ", "NO"],
        "Pain_Migration": ["No", "Yes", "no", "YES"],
    })

    out = encode_boolean_cols(df, columns=["Is_Pregnant", "Pain_Migration"])

    assert out["Is_Pregnant"].tolist() == [1, 0, 1, 0]
    assert out["Pain_Migration"].tolist() == [0, 1, 0, 1]


def test_processed_output_has_no_nulls():
    df_raw = pd.read_csv("data/raw/appendicitis_comprehensive_dataset.csv")
    df_clean = clean_data(df_raw)

    X, y = split_X_y(df_clean)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    X_train_df = pd.DataFrame(
        X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t,
        columns=preprocessor.get_feature_names_out()
    )
    X_test_df = pd.DataFrame(
        X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t,
        columns=preprocessor.get_feature_names_out()
    )

    assert X_train_df.isna().sum().sum() == 0
    assert X_test_df.isna().sum().sum() == 0