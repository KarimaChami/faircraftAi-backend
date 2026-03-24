import joblib
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

# Path to the serialized model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../src/ml/models/price_model.joblib"
)

# Resolve to an absolute path for safety
MODEL_PATH = os.path.abspath(MODEL_PATH)

_model_instance = None

def get_model():
    """
    Dependency to return the pre-loaded ML model.
    Loads it once into memory on the first request if not loaded.
    """
    global _model_instance
    if _model_instance is None:
        try:
            logger.info(f"Loading ML model from {MODEL_PATH}")
            _model_instance = joblib.load(MODEL_PATH)
            logger.info("ML model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            raise RuntimeError(f"Could not load ML model: {e}")
    return _model_instance
