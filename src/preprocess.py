import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)


def split_features_target(
    df: pd.DataFrame, target_column: str = "Class"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataset into features (X) and target (y)."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def scale_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], StandardScaler]:
    """
    Standardize 'Amount' and 'Time' columns using StandardScaler.
    Fit on X_train, transform both X_train and X_test.
    """
    cols_to_scale = [col for col in ["Amount", "Time"] if col in X_train.columns]

    X_train_scaled = X_train.copy()
    if scaler is None:
        scaler = StandardScaler()
        if cols_to_scale:
            X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    else:
        if cols_to_scale:
            X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = X_test.copy()
        if cols_to_scale:
            X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train_scaled, X_test_scaled, scaler
