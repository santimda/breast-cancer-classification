import numpy as np
import pandas as pd


QUANTITIES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave points",
    "symmetry",
    "fractal dimension",
]


def build_final_features(measurements: pd.DataFrame) -> pd.DataFrame:
    """Build the 20 feature columns used by the final classifier."""
    mean_features = [f"mean {quantity}" for quantity in QUANTITIES]
    features = measurements[mean_features].copy()

    for quantity in QUANTITIES:
        mean = measurements[f"mean {quantity}"]
        worst = measurements[f"worst {quantity}"]
        features[f"{quantity} worst/mean"] = np.where(mean == 0, 0, worst / mean)

    return features
