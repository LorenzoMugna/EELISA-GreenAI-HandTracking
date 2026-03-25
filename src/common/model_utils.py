"""Shared utilities for XGBoost model loading and prediction."""

from pathlib import Path


# Message type constants to replace string literals
class MessageTypes:
    """Constants for UDP message types."""
    SPIKE = "spike"
    VALUE = "value"
    COORDINATE = "coordinate"
    MODEL = "model"


# Configuration constants
class Config:
    """Common configuration constants."""
    DEFAULT_MODEL_PATH = "PredictionModelNoSpike.json"
    DEFAULT_SCALER_PATH = "scalerNoSpike.pkl"
    DEFAULT_IMAGE_SIZE = (200, 200)
    DEFAULT_UPDATE_INTERVAL_MS = 500
    DEFAULT_SPIKE_TIMELINE_SECONDS = 5


def load_model_and_scaler(model_path: str = "PredictionModelNoSpike.json",
                         scaler_path: str = "scalerNoSpike.pkl"):
    """Load XGBoost model and scaler with error handling.

    Args:
        model_path: Path to the XGBoost model file
        scaler_path: Path to the scaler pickle file

    Returns:
        tuple: (loaded_model, scaler) or (None, None) if failed
    """
    # Import here to avoid dependency issues when only using constants
    import joblib
    import xgboost as xgb

    try:
        # Load XGBoost model
        loaded_model = xgb.Booster()
        loaded_model.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None

    try:
        # Load scaler
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded successfully from {scaler_path}")
    except Exception as e:
        print(f"Error loading scaler from {scaler_path}: {e}")
        return loaded_model, None

    return loaded_model, scaler


def create_model_predictor(model, scaler):
    """Create a model predictor function.

    Args:
        model: Trained XGBoost model
        scaler: Fitted scaler for data preprocessing

    Returns:
        Function that takes data and returns (prediction, probabilities)
    """
    # Import here to avoid dependency issues
    import numpy as np
    import xgboost as xgb

    def predict(data):
        try:
            # Convert to numpy array and reshape
            data_array = np.array(data, dtype=np.float32).reshape(1, -1)

            # Scale the data if scaler available
            if scaler:
                data_array = scaler.transform(data_array)

            # Create DMatrix and predict
            dmatrix = xgb.DMatrix(data_array)
            probabilities = model.predict(dmatrix)[0]  # Get first (and only) prediction
            prediction = np.argmax(probabilities)

            return int(prediction), probabilities
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return predict


# Message type constants to replace string literals
class MessageTypes:
    """Constants for UDP message types."""
    SPIKE = "spike"
    VALUE = "value"
    COORDINATE = "coordinate"
    MODEL = "model"


# Configuration constants
class Config:
    """Common configuration constants."""
    DEFAULT_MODEL_PATH = "PredictionModelNoSpike.json"
    DEFAULT_SCALER_PATH = "scalerNoSpike.pkl"
    DEFAULT_IMAGE_SIZE = (200, 200)
    DEFAULT_UPDATE_INTERVAL_MS = 500
    DEFAULT_SPIKE_TIMELINE_SECONDS = 5