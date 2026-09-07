from fastapi.testclient import TestClient

from api.app import EXAMPLE_MEASUREMENTS, app
from predict import predict_one


client = TestClient(app)


def test_predict_one_returns_prediction_payload():
    result = predict_one(EXAMPLE_MEASUREMENTS)

    assert set(result) == {
        "prediction",
        "malignant_probability",
        "threshold",
    }
    assert result["prediction"] in (0, 1)
    assert 0 <= result["malignant_probability"] <= 1
    assert result["threshold"] == 0.225


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_with_valid_measurements():
    response = client.post(
        "/predict",
        json={"measurements": EXAMPLE_MEASUREMENTS},
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["threshold"] == 0.225


def test_predict_endpoint_rejects_incomplete_measurements():
    response = client.post(
        "/predict",
        json={"measurements": {"mean radius": 17.99}},
    )

    assert response.status_code == 422
