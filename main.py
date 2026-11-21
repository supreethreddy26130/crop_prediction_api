from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "random_forest_model.pkl")
feature_names = list(joblib.load(BASE_DIR / "feature_names.pkl"))

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
FEATURES_PATH = BASE_DIR / "feature_names.pkl"

with MODEL_PATH.open("rb") as f:
    model = joblib.load(f)

with FEATURES_PATH.open("rb") as f:
    feature_names = joblib.load(f)

feature_names = list(feature_names)

app = FastAPI(title="Crop Recommendation API")

class CropFeatures(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class PredictionRequest(BaseModel):
    features: CropFeatures

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "N": 90,
                    "P": 42,
                    "K": 43,
                    "temperature": 20.9,
                    "humidity": 82.0,
                    "ph": 6.5,
                    "rainfall": 202.9,
                }
            }
        }
    }


@app.get("/")
def home():
    return {"message": "Random Forest API is running"}


@app.get("/features")
def get_features():
    return {"features": feature_names}


@app.post("/predict")
def predict(request: PredictionRequest):
    supplied = request.features.model_dump()
    missing = [name for name in feature_names if name not in supplied]
    extra = [name for name in supplied if name not in feature_names]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": extra},
        )

    row = {name: supplied[name] for name in feature_names}
    input_data = pd.DataFrame([row], columns=feature_names)

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    return {
        "predicted_crop": str(prediction),
        "probabilities": [
            {
                "class": str(crop),
                "probability": round(float(probability), 4),
            }
            for crop, probability in zip(model.classes_, probabilities)
        ],
    }
