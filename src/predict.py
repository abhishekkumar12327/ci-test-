import pandas as pd
from typing import List, Union
from src.model import load_artifacts
from src.preprocess import scale_features


def predict_fraud(
    model_path: str, data: Union[pd.DataFrame, List[dict]]
) -> List[int]:
    """
    Run fraud prediction on incoming feature data.
    0 = Normal transaction, 1 = Fraudulent transaction.
    """
    if isinstance(data, list):
        data = pd.DataFrame(data)

    artifacts = load_artifacts(model_path)
    model = artifacts["model"]
    scaler = artifacts["scaler"]

    scaled_data, _, _ = scale_features(data, scaler=scaler)
    predictions = model.predict(scaled_data)
    return predictions.tolist()
