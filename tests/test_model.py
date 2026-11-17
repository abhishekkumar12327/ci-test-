import os
import tempfile
from src.preprocess import load_data, split_features_target, scale_features
from src.model import train_model, evaluate_model, save_artifacts, load_artifacts
from src.predict import predict_fraud

SAMPLE_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "sample_creditcard.csv"
)


def test_model_training_and_evaluation():
    """Verify model trains, evaluates and produces valid metrics."""
    df = load_data(SAMPLE_CSV_PATH)
    X, y = split_features_target(df)
    X_train_raw, X_test_raw = X.iloc[:90], X.iloc[90:]
    y_train, y_test = y.iloc[:90], y.iloc[90:]

    X_train, X_test, _ = scale_features(X_train_raw, X_test_raw)
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_model_save_load_predict():
    """Verify saving artifacts, loading and generating predictions."""
    df = load_data(SAMPLE_CSV_PATH)
    X, y = split_features_target(df)
    X_train_raw = X.iloc[:100]
    y_train = y.iloc[:100]

    X_train, _, scaler = scale_features(X_train_raw)
    model = train_model(X_train, y_train)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_artifacts(model, scaler, tmp_path)
        artifacts = load_artifacts(tmp_path)
        assert "model" in artifacts
        assert "scaler" in artifacts

        # Predict with predict_fraud
        sample_batch = X.iloc[100:105]
        preds = predict_fraud(tmp_path, sample_batch)
        assert len(preds) == len(sample_batch)
        assert all(p in [0, 1] for p in preds)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
