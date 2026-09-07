from collections.abc import Mapping
import json
from pathlib import Path

import joblib
import pandas as pd

from features import QUANTITIES, build_final_features


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "model.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "model" / "threshold.json"

RAW_COLUMNS = (
    [f"mean {quantity}" for quantity in QUANTITIES]
    + [f"{quantity} error" for quantity in QUANTITIES]
    + [f"worst {quantity}" for quantity in QUANTITIES]
)


def _prepare_measurements(raw_measurements):
    if isinstance(raw_measurements, Mapping):
        measurements = pd.DataFrame([dict(raw_measurements)])
    elif isinstance(raw_measurements, pd.DataFrame):
        measurements = raw_measurements.copy()
    else:
        raise TypeError("raw_measurements must be a mapping or a pandas DataFrame")

    if len(measurements) != 1:
        raise ValueError("Prediction accepts exactly one observation")

    missing_columns = sorted(set(RAW_COLUMNS) - set(measurements.columns))
    if missing_columns:
        raise ValueError(f"Missing raw measurement columns: {missing_columns}")

    return measurements


def predict_one(raw_measurements):
    """Return the malignant probability and thresholded prediction."""
    measurements = _prepare_measurements(raw_measurements)
    features = build_final_features(measurements)

    model = joblib.load(MODEL_PATH)
    threshold_metadata = json.loads(THRESHOLD_PATH.read_text())
    threshold = float(threshold_metadata["threshold"])

    probabilities = model.predict_proba(features)[0]
    positive_index = list(model.classes_).index(1)
    malignant_probability = float(probabilities[positive_index])

    return {
        "prediction": int(malignant_probability >= threshold),
        "malignant_probability": malignant_probability,
        "threshold": threshold,
    }
