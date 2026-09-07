import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import recall_score

from train import build_voting_classifier, load_final_features


SPECIFICITY_CONSTRAINT = 0.90
N_SPLITS = 5
RANDOM_STATE = 42


def select_threshold(probabilities, target):
    threshold_results = []

    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        sensitivity = recall_score(target, predictions, pos_label=1)
        specificity = recall_score(target, predictions, pos_label=0)

        if specificity >= SPECIFICITY_CONSTRAINT:
            threshold_results.append({
                "threshold": float(round(threshold, 3)),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "false_negative_rate": float(1 - sensitivity),
            })

    if not threshold_results:
        raise ValueError(
            "No candidate threshold satisfies the specificity constraint."
        )

    return max(
        threshold_results,
        key=lambda result: (result["sensitivity"], result["specificity"]),
    )


def main():
    features, target = load_final_features()
    model = build_voting_classifier()
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Out-of-fold probabilities prevent threshold selection from using
    # predictions made on observations used to fit each fold's model.
    oof_probabilities = cross_val_predict(
        model,
        features,
        target,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    selected = select_threshold(oof_probabilities, target)
    threshold_path = Path(__file__).resolve().parent / "model" / "threshold.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "threshold": selected["threshold"],
        "sensitivity": selected["sensitivity"],
        "specificity": selected["specificity"],
        "false_negative_rate": selected["false_negative_rate"],
        "specificity_constraint": SPECIFICITY_CONSTRAINT,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "n_observations": len(features),
        "n_features": features.shape[1],
        "positive_class": "malignant (1)",
    }

    threshold_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Saved threshold metadata to {threshold_path}")


if __name__ == "__main__":
    main()
