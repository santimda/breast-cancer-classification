## Project

`breast-cancer-classifier` is an educational ML project comparing classifiers on the Wisconsin Diagnostic Breast Cancer dataset.

The analysis notebook is the reference for the current results and methodology.

### Current final model

- Target: `0 = benign`, `1 = malignant`
- Features: 10 mean measurements + 10 worst/mean ratios
- Classifier: weighted soft-voting ensemble
- Base models: Logistic Regression, SVM, Random Forest, Gradient Boosting, KNN
- Final decision threshold: `0.220`
- Test results at this threshold:
  - Accuracy: 97.4%
  - Sensitivity/recall: 97.6%
  - Specificity: 97.2%
  - ROC-AUC: 99.9%

The threshold is an operating parameter applied to the predicted malignant probability; it is not part of the classifier itself.

## Goal

Turn the final notebook model into a small, reproducible ML application that can be trained, tested, and served through an API.

Keep the implementation simple. Do not add unnecessary infrastructure.

## Implementation plan

Work **one step at a time**. After each step:

1. Implement only that step.
2. Run/check it.
3. Explain what was built and why.
4. Stop and wait for confirmation before continuing.

### Step 1 — Training script

Create `train.py` containing only:

1. Load the dataset.
2. Construct the final 20 features.
3. Construct the final Voting classifier.
4. Fit it on all available labeled observations.
5. Save the trained classifier to `model/model.joblib`.

Do **not** include EDA, plots, cross-validation, threshold selection, feature-selection experiments, or model comparisons. Use the best-fit hyperparameters and Voting weights already obtained in the ipynb instead of re-fitting them. The deployment model is trained on all available labeled data because it will not be independently re-tested after deployment training.

### Step 2 — Reusable feature pipeline

Move feature engineering into reusable code and update `train.py` to use it, so training and inference use exactly the same transformations.

The shared code should construct the ten mean features and ten `worst / mean` ratio features with the same zero-denominator handling as the notebook.

### Step 3 — Threshold calibration

Create `calibrate_threshold.py` containing only the threshold-calibration workflow:

1. Load the labeled dataset and the reusable feature transformation.
2. Recreate the frozen Voting classifier configuration.
3. Generate out-of-fold malignant probabilities across all observations.
4. Select the threshold that maximizes sensitivity subject to at least 90% specificity.
5. Save the threshold and calibration metadata to `model/threshold.json`.

Do not use the fitted all-data model's in-sample predictions for threshold selection. The threshold is an operating parameter, so save it separately from the classifier.

### Step 4 — Prediction module

Create `predict.py` that accepts raw measurements, applies the same transformation, loads the model, returns malignant probability, loads the calibrated threshold from `model/threshold.json`, and returns the prediction.

The initial notebook threshold is `0.220`; the calibrated value produced by Step 3 is authoritative and must not be hardcoded in the prediction module.

### Step 5 — API

Create a small FastAPI application with:

- `GET /health`
- `POST /predict`

The API should return the prediction, malignant probability, and threshold.

### Step 6 — Tests

Add basic `pytest` tests for prediction logic and the API.

### Step 7 — Docker

Add a minimal `Dockerfile` so the API can be built and run reproducibly.

### Optional Step 8 — CI

Add a simple GitHub Actions workflow to run the tests automatically.

## Constraints

- Preserve the current final model and feature representation unless explicitly discussed.
- Keep the notebook as the analysis record; scripts contain the production-style implementation.
- Avoid data leakage.
- Keep training and inference transformations identical.
- Prefer small, readable functions.
- Use explicit random seeds where relevant.
- Do not introduce Kubernetes, cloud deployment, Terraform, Airflow, Kafka, or similar infrastructure.

## Target structure

```text
breast-cancer-classifier/
├── breast_cancer_classifier_comparison.ipynb
├── train.py
├── calibrate_threshold.py
├── predict.py
├── api/
│   └── app.py
├── tests/
│   └── test_api.py
├── model/
│   ├── model.joblib
│   └── threshold.json
├── figs/
├── Dockerfile
├── requirements.txt
├── README.md
└── AGENT.md
