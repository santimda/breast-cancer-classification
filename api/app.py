from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from predict import predict_one


app = FastAPI(
    title="Breast Cancer Classifier API",
    description="Predicts the probability that a breast mass is malignant.",
    version="1.0.0",
)


class Measurements(BaseModel):
    """The 30 raw measurements expected by the model."""

    model_config = ConfigDict(populate_by_name=True)

    mean_radius: float = Field(alias="mean radius")
    mean_texture: float = Field(alias="mean texture")
    mean_perimeter: float = Field(alias="mean perimeter")
    mean_area: float = Field(alias="mean area")
    mean_smoothness: float = Field(alias="mean smoothness")
    mean_compactness: float = Field(alias="mean compactness")
    mean_concavity: float = Field(alias="mean concavity")
    mean_concave_points: float = Field(alias="mean concave points")
    mean_symmetry: float = Field(alias="mean symmetry")
    mean_fractal_dimension: float = Field(alias="mean fractal dimension")
    radius_error: float = Field(alias="radius error")
    texture_error: float = Field(alias="texture error")
    perimeter_error: float = Field(alias="perimeter error")
    area_error: float = Field(alias="area error")
    smoothness_error: float = Field(alias="smoothness error")
    compactness_error: float = Field(alias="compactness error")
    concavity_error: float = Field(alias="concavity error")
    concave_points_error: float = Field(alias="concave points error")
    symmetry_error: float = Field(alias="symmetry error")
    fractal_dimension_error: float = Field(alias="fractal dimension error")
    worst_radius: float = Field(alias="worst radius")
    worst_texture: float = Field(alias="worst texture")
    worst_perimeter: float = Field(alias="worst perimeter")
    worst_area: float = Field(alias="worst area")
    worst_smoothness: float = Field(alias="worst smoothness")
    worst_compactness: float = Field(alias="worst compactness")
    worst_concavity: float = Field(alias="worst concavity")
    worst_concave_points: float = Field(alias="worst concave points")
    worst_symmetry: float = Field(alias="worst symmetry")
    worst_fractal_dimension: float = Field(alias="worst fractal dimension")


EXAMPLE_MEASUREMENTS = {
    "mean radius": 17.99,
    "mean texture": 10.38,
    "mean perimeter": 122.8,
    "mean area": 1001.0,
    "mean smoothness": 0.1184,
    "mean compactness": 0.2776,
    "mean concavity": 0.3001,
    "mean concave points": 0.1471,
    "mean symmetry": 0.2419,
    "mean fractal dimension": 0.07871,
    "radius error": 1.095,
    "texture error": 0.9053,
    "perimeter error": 8.589,
    "area error": 153.4,
    "smoothness error": 0.006399,
    "compactness error": 0.04904,
    "concavity error": 0.05373,
    "concave points error": 0.01587,
    "symmetry error": 0.03003,
    "fractal dimension error": 0.006193,
    "worst radius": 25.38,
    "worst texture": 17.33,
    "worst perimeter": 184.6,
    "worst area": 2019.0,
    "worst smoothness": 0.1622,
    "worst compactness": 0.6656,
    "worst concavity": 0.7119,
    "worst concave points": 0.2654,
    "worst symmetry": 0.4601,
    "worst fractal dimension": 0.1189,
}


class PredictionRequest(BaseModel):
    """Raw measurements for one observation.

    The example is the first observation in the bundled dataset. With the
    current model artifact it returns a malignant probability of approximately
    0.986, threshold 0.225, and prediction 1. This is an example, not a
    guaranteed output for future model versions.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"measurements": EXAMPLE_MEASUREMENTS}
        }
    )

    measurements: Measurements


class PredictionResponse(BaseModel):
    prediction: int
    malignant_probability: float
    threshold: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        return predict_one(request.measurements.model_dump(by_alias=True))
    except (TypeError, ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
