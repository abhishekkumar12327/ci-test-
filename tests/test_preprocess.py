import os
import pandas as pd
import pytest
from src.preprocess import load_data, split_features_target, scale_features

SAMPLE_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "sample_creditcard.csv"
)


def test_load_data():
    """Verify sample dataset loads and is not empty."""
    df = load_data(SAMPLE_CSV_PATH)
    assert not df.empty
    assert "Class" in df.columns
    assert "Amount" in df.columns
    assert "Time" in df.columns


def test_split_features_target():
    """Verify feature/target separation."""
    df = load_data(SAMPLE_CSV_PATH)
    X, y = split_features_target(df, target_column="Class")
    assert "Class" not in X.columns
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})


def test_split_features_invalid_column():
    """Verify proper exception on missing target column."""
    df = pd.DataFrame({"colA": [1, 2], "colB": [3, 4]})
    with pytest.raises(ValueError):
        split_features_target(df, target_column="NonExistent")


def test_scale_features():
    """Verify standard scaling on Amount and Time."""
    df = load_data(SAMPLE_CSV_PATH)
    X, _ = split_features_target(df)
    X_train = X.iloc[:80]
    X_test = X.iloc[80:]

    scaled_train, scaled_test, scaler = scale_features(X_train, X_test)
    assert scaled_train is not None
    assert scaled_test is not None
    assert scaler is not None
    # Mean of scaled train features should be close to 0
    assert abs(scaled_train["Amount"].mean()) < 1e-5
    assert abs(scaled_train["Time"].mean()) < 1e-5
