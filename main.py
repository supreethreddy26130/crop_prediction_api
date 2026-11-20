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

# Only load pickle files from a trusted source.
with MODEL_PATH.open("rb") as f:
    model = joblib.load(f)

with FEATURES_PATH.open("rb") as f:
    feature_names = joblib.load(f)

# Handle feature names stored as a list, tuple, or NumPy array.
feature_names = list(feature_names)

app = FastAPI(title="Crop Recommendation API")

class PredictionRequest(BaseModel):
    features: dict[str, float]


@app.get("/")
def home():
    return {"message": "Random Forest API is running"}


@app.get("/features")
def get_features():
    return {"features": feature_names}


@app.post("/predict")
def predict(request: PredictionRequest):
    supplied = request.features
    missing = [name for name in feature_names if name not in supplied]
    extra = [name for name in supplied if name not in feature_names]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": extra},
        )

    # Explicitly order inputs to match the model's training feature order.
    row = {name: supplied[name] for name in feature_names}
    input_data = pd.DataFrame([row], columns=feature_names)

    prediction = model.predict(input_data)[0]
    result = {"prediction": prediction.item() if hasattr(prediction, "item") else prediction}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        result["probabilities"] = [
            float(value) for value in probabilities
        ]

        if hasattr(model, "classes_"):
            result["classes"] = [
                value.item() if hasattr(value, "item") else value
                for value in model.classes_
            ]

    return result