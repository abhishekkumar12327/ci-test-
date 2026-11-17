import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any


def train_model(X_train, y_train, var_smoothing: float = 1e-9) -> GaussianNB:
    """Train a Gaussian Naive Bayes model."""
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: GaussianNB, X_test, y_test) -> Dict[str, float]:
    """Evaluate model performance returning standard metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    return metrics


def save_artifacts(model: GaussianNB, scaler: Any, filepath: str) -> None:
    """Save trained model and scaler to a file."""
    joblib.dump({"model": model, "scaler": scaler}, filepath)


def load_artifacts(filepath: str) -> Dict[str, Any]:
    """Load model and scaler from a file."""
    return joblib.load(filepath)
