from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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


def load_final_features():
    data = load_breast_cancer()
    measurements = pd.DataFrame(data.data, columns=data.feature_names)

    features = measurements[[f"mean {quantity}" for quantity in QUANTITIES]].copy()
    for quantity in QUANTITIES:
        mean = measurements[f"mean {quantity}"]
        worst = measurements[f"worst {quantity}"]
        features[f"{quantity} worst/mean"] = np.where(mean == 0, 0, worst / mean)

    target = pd.Series((data.target == 0).astype(int), name="target")
    return features, target


def build_voting_classifier():
    estimators = [
        (
            "Logistic Regression",
            Pipeline([
                ("preprocessor", StandardScaler()),
                ("classifier", LogisticRegression(
                    C=1.5,
                    max_iter=5000,
                    random_state=42,
                )),
            ]),
        ),
        (
            "SVM",
            Pipeline([
                ("preprocessor", StandardScaler()),
                ("classifier", SVC(
                    C=2,
                    gamma=0.03,
                    probability=True,
                    random_state=42,
                )),
            ]),
        ),
        (
            "Random Forest",
            Pipeline([
                ("preprocessor", "passthrough"),
                ("classifier", RandomForestClassifier(
                    n_estimators=180,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]),
        ),
        (
            "Gradient Boosting",
            Pipeline([
                ("preprocessor", "passthrough"),
                ("classifier", GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.15,
                    max_depth=2,
                    random_state=42,
                )),
            ]),
        ),
        (
            "K-Nearest Neighbors",
            Pipeline([
                ("preprocessor", StandardScaler()),
                ("classifier", KNeighborsClassifier(
                    n_neighbors=11,
                    weights="distance",
                    metric="euclidean",
                )),
            ]),
        ),
    ]

    # Fixed weights from the notebook's ROC-AUC-based weighting rule.
    weights = [
        0.19185367,
        0.24103904,
        0.18339994,
        0.23642791,
        0.14727943,
    ]

    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=weights,
    )


def main():
    features, target = load_final_features()
    model = build_voting_classifier()
    model.fit(features, target)

    model_path = Path(__file__).resolve().parent / "model" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
