from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json
import sys
from typing import List

# ---------------------------------------------------
# App Configuration
# ---------------------------------------------------

app = FastAPI(
    title="Faminga Yield Prediction API",
    description="Predict bean and maize yields based on soil, weather, and crop data",
    version="1.0.0"
)

# CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Load Model & Preprocessing Objects
# ---------------------------------------------------

try:
    print(f"🐍 Python version: {sys.version}")
    print("📦 Loading ML artifacts...")

    model = joblib.load("crop_yield_model_gradient_boosting.pkl")
    scaler = joblib.load("scaler.pkl")

    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)

    print("✅ Model, scaler, and feature names loaded successfully")

except Exception as e:
    print("❌ Failed to load ML artifacts")
    print(f"📍 Error: {e}")
    raise RuntimeError("Model loading failed. Check deployment logs.") from e


# ---------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------

class YieldPredictionInput(BaseModel):
    soil_ph: float = Field(..., ge=4.0, le=9.0)
    nitrogen_ppm: float = Field(..., ge=0, le=200)
    phosphorus_ppm: float = Field(..., ge=0, le=100)
    potassium_ppm: float = Field(..., ge=0, le=400)
    soil_moisture: float = Field(..., ge=0, le=100)
    organic_matter: float = Field(..., ge=0, le=15)
    total_rainfall_mm: float = Field(..., ge=0, le=2000)
    avg_temperature_c: float = Field(..., ge=0, le=50)
    avg_humidity: float = Field(..., ge=0, le=100)
    solar_radiation: float = Field(..., ge=0, le=30)
    crop_type: str = Field(..., description="beans or maize")
    planting_month: int = Field(..., ge=1, le=12)
    field_size_ha: float = Field(..., ge=0.1, le=10)
    fertilizer_kg_per_ha: float = Field(..., ge=0, le=300)
    irrigation: int = Field(..., ge=0, le=1)


class YieldPredictionOutput(BaseModel):
    predicted_yield_kg_per_ha: float
    crop_type: str
    confidence: str
    recommendations: List[str]


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------

def preprocess_input(data: YieldPredictionInput) -> np.ndarray:
    """Convert request data into scaled feature vector"""

    features = {
        "soil_ph": data.soil_ph,
        "nitrogen_ppm": data.nitrogen_ppm,
        "phosphorus_ppm": data.phosphorus_ppm,
        "potassium_ppm": data.potassium_ppm,
        "soil_moisture": data.soil_moisture,
        "organic_matter": data.organic_matter,
        "total_rainfall_mm": data.total_rainfall_mm,
        "avg_temperature_c": data.avg_temperature_c,
        "avg_humidity": data.avg_humidity,
        "solar_radiation": data.solar_radiation,
        "planting_month": data.planting_month,
        "field_size_ha": data.field_size_ha,
        "fertilizer_kg_per_ha": data.fertilizer_kg_per_ha,
        "irrigation": data.irrigation,

        # Engineered features
        "npk_ratio": (data.nitrogen_ppm + data.phosphorus_ppm + data.potassium_ppm) / 3,
        "rainfall_per_temp": (
            data.total_rainfall_mm / data.avg_temperature_c
            if data.avg_temperature_c > 0 else 0
        ),
        "nutrient_efficiency": 0,

        # One-hot encoding
        "crop_type_beans": 1 if data.crop_type.lower() == "beans" else 0,
        "crop_type_maize": 1 if data.crop_type.lower() == "maize" else 0,
    }

    ordered_features = np.array([[features.get(f, 0) for f in feature_names]])
    return scaler.transform(ordered_features)


def generate_recommendations(prediction: float, data: YieldPredictionInput) -> List[str]:
    recommendations = []

    if prediction < 3000:
        recommendations.append("⚠️ Low yield predicted. Improve soil nutrients and water management.")
    elif prediction > 5500:
        recommendations.append("✅ High yield potential. Maintain current farming practices.")
    else:
        recommendations.append("📊 Moderate yield expected. Optimization is possible.")

    if data.soil_ph < 5.5:
        recommendations.append("🌱 Soil is acidic. Consider lime application.")
    elif data.soil_ph > 7.5:
        recommendations.append("🌱 Soil is alkaline. Monitor nutrient availability.")

    if data.nitrogen_ppm < 50:
        recommendations.append("💧 Nitrogen is low. Apply nitrogen-rich fertilizer.")
    if data.phosphorus_ppm < 20:
        recommendations.append("💧 Phosphorus is low. Apply phosphate fertilizer.")
    if data.potassium_ppm < 150:
        recommendations.append("💧 Potassium is low. Apply potash fertilizer.")

    if data.irrigation == 0 and data.total_rainfall_mm < 600:
        recommendations.append("💦 Rainfall is low. Consider irrigation.")

    return recommendations


# ---------------------------------------------------
# API Endpoints
# ---------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "active",
        "service": "Faminga Yield Prediction API",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_count": len(feature_names)
    }


@app.post("/predict", response_model=YieldPredictionOutput)
def predict_yield(data: YieldPredictionInput):

    if data.crop_type.lower() not in ["beans", "maize"]:
        raise HTTPException(
            status_code=400,
            detail="crop_type must be 'beans' or 'maize'"
        )

    try:
        processed = preprocess_input(data)
        prediction = float(model.predict(processed)[0])
        prediction = max(0, prediction)

        recommendations = generate_recommendations(prediction, data)
        confidence = "high" if data.irrigation == 1 and data.fertilizer_kg_per_ha > 80 else "moderate"

        return YieldPredictionOutput(
            predicted_yield_kg_per_ha=round(prediction, 2),
            crop_type=data.crop_type,
            confidence=confidence,
            recommendations=recommendations
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ---------------------------------------------------
# Local Development Entry Point
# ---------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
