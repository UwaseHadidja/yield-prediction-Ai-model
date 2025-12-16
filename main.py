from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json
from typing import Optional
import sys
import pickle

app = FastAPI(
    title="Faminga Yield Prediction API",
    description="Predict bean and maize yields based on soil, weather, and crop data",
    version="1.0.0"
)

# CORS middleware for your Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and preprocessing objects with compatibility handling
try:
    print(f"🐍 Python version: {sys.version}")
    print("📦 Loading models...")

    # Try loading with joblib first
    try:
        model = joblib.load("crop_yield_model_gradient_boosting.pkl")
        scaler = joblib.load("scaler.pkl")
    except ModuleNotFoundError as e:
        print(f"⚠️ Joblib loading failed: {e}")
        print("🔄 Trying alternative loading method with pickle...")

        # Fallback to standard pickle with custom unpickler
        import importlib

        class CompatibleUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Handle sklearn internal modules that may have moved
                if module.startswith('sklearn.ensemble._gb_losses'):
                    module = 'sklearn.ensemble._gb'
                elif '_loss' in module:
                    # Map old loss modules to new ones
                    module = module.replace('._loss', '.ensemble._gb')

                try:
                    return super().find_class(module, name)
                except (ModuleNotFoundError, AttributeError):
                    # Try importing from sklearn.ensemble._gb
                    if 'loss' in name.lower() or 'loss' in module.lower():
                        try:
                            mod = importlib.import_module('sklearn.ensemble._gb')
                            return getattr(mod, name)
                        except:
                            pass
                    raise

        with open("crop_yield_model_gradient_boosting.pkl", "rb") as f:
            model = CompatibleUnpickler(f).load()

        with open("scaler.pkl", "rb") as f:
            scaler = CompatibleUnpickler(f).load()

    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)

    print("✅ Models loaded successfully")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    print(f"📍 Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    raise

class YieldPredictionInput(BaseModel):
    """Input schema for yield prediction"""
    soil_ph: float = Field(..., ge=4.0, le=9.0, description="Soil pH level (4-9)")
    nitrogen_ppm: float = Field(..., ge=0, le=200, description="Nitrogen content (ppm)")
    phosphorus_ppm: float = Field(..., ge=0, le=100, description="Phosphorus content (ppm)")
    potassium_ppm: float = Field(..., ge=0, le=400, description="Potassium content (ppm)")
    soil_moisture: float = Field(..., ge=0, le=100, description="Soil moisture percentage")
    organic_matter: float = Field(..., ge=0, le=15, description="Organic matter percentage")
    total_rainfall_mm: float = Field(..., ge=0, le=2000, description="Total rainfall (mm)")
    avg_temperature_c: float = Field(..., ge=0, le=50, description="Average temperature (°C)")
    avg_humidity: float = Field(..., ge=0, le=100, description="Average humidity (%)")
    solar_radiation: float = Field(..., ge=0, le=30, description="Solar radiation")
    crop_type: str = Field(..., description="Crop type: 'beans' or 'maize'")
    planting_month: int = Field(..., ge=1, le=12, description="Planting month (1-12)")
    field_size_ha: float = Field(..., ge=0.1, le=10, description="Field size (hectares)")
    fertilizer_kg_per_ha: float = Field(..., ge=0, le=300, description="Fertilizer amount (kg/ha)")
    irrigation: int = Field(..., ge=0, le=1, description="Irrigation: 0 (no) or 1 (yes)")

    class Config:
        json_schema_extra = {
            "example": {
                "soil_ph": 6.5,
                "nitrogen_ppm": 75.0,
                "phosphorus_ppm": 30.0,
                "potassium_ppm": 200.0,
                "soil_moisture": 28.0,
                "organic_matter": 5.5,
                "total_rainfall_mm": 850.0,
                "avg_temperature_c": 22.0,
                "avg_humidity": 75.0,
                "solar_radiation": 20.0,
                "crop_type": "maize",
                "planting_month": 3,
                "field_size_ha": 3.5,
                "fertilizer_kg_per_ha": 120.0,
                "irrigation": 1
            }
        }

class YieldPredictionOutput(BaseModel):
    """Output schema for yield prediction"""
    predicted_yield_kg_per_ha: float
    crop_type: str
    confidence: str
    recommendations: list[str]

def preprocess_input(data: YieldPredictionInput) -> np.ndarray:
    """Preprocess input data to match training format"""
    # Create base features
    features = {
        'soil_ph': data.soil_ph,
        'nitrogen_ppm': data.nitrogen_ppm,
        'phosphorus_ppm': data.phosphorus_ppm,
        'potassium_ppm': data.potassium_ppm,
        'soil_moisture': data.soil_moisture,
        'organic_matter': data.organic_matter,
        'total_rainfall_mm': data.total_rainfall_mm,
        'avg_temperature_c': data.avg_temperature_c,
        'avg_humidity': data.avg_humidity,
        'solar_radiation': data.solar_radiation,
        'planting_month': data.planting_month,
        'field_size_ha': data.field_size_ha,
        'fertilizer_kg_per_ha': data.fertilizer_kg_per_ha,
        'irrigation': data.irrigation
    }

    # Feature engineering (matching notebook preprocessing)
    features['npk_ratio'] = (data.nitrogen_ppm + data.phosphorus_ppm + data.potassium_ppm) / 3
    features['rainfall_per_temp'] = data.total_rainfall_mm / data.avg_temperature_c if data.avg_temperature_c > 0 else 0
    features['nutrient_efficiency'] = 0  # This is calculated post-prediction in training

    # One-hot encode crop_type
    features['crop_type_beans'] = 1 if data.crop_type.lower() == 'beans' else 0
    features['crop_type_maize'] = 1 if data.crop_type.lower() == 'maize' else 0

    # Convert to array in correct order
    feature_array = np.array([[features.get(name, 0) for name in feature_names]])

    # Scale features
    scaled_features = scaler.transform(feature_array)

    return scaled_features

def generate_recommendations(prediction: float, data: YieldPredictionInput) -> list[str]:
    """Generate farming recommendations based on prediction and input"""
    recommendations = []

    if prediction < 3000:
        recommendations.append("⚠️ Low yield predicted. Consider soil testing and nutrient supplementation.")
    elif prediction > 5500:
        recommendations.append("✅ High yield potential! Maintain current practices.")
    else:
        recommendations.append("📊 Moderate yield expected. Room for optimization.")

    # Soil recommendations
    if data.soil_ph < 5.5:
        recommendations.append("🌱 Soil is acidic. Consider lime application to raise pH.")
    elif data.soil_ph > 7.5:
        recommendations.append("🌱 Soil is alkaline. Monitor nutrient availability.")

    # Fertilizer recommendations
    if data.nitrogen_ppm < 50:
        recommendations.append("💧 Low nitrogen levels. Apply nitrogen-rich fertilizer.")
    if data.phosphorus_ppm < 20:
        recommendations.append("💧 Low phosphorus. Consider phosphate fertilizer.")
    if data.potassium_ppm < 150:
        recommendations.append("💧 Low potassium. Apply potash fertilizer.")

    # Irrigation recommendations
    if data.irrigation == 0 and data.total_rainfall_mm < 600:
        recommendations.append("💦 Low rainfall detected. Consider installing irrigation system.")

    return recommendations

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "service": "Faminga Yield Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "feature_count": len(feature_names)
    }

@app.post("/predict", response_model=YieldPredictionOutput)
def predict_yield(data: YieldPredictionInput):
    """
    Predict crop yield based on input parameters

    Returns predicted yield in kg/ha along with recommendations
    """
    try:
        # Validate crop type
        if data.crop_type.lower() not in ['beans', 'maize']:
            raise HTTPException(status_code=400, detail="crop_type must be 'beans' or 'maize'")

        # Preprocess input
        processed_data = preprocess_input(data)

        # Make prediction
        prediction = model.predict(processed_data)[0]
        prediction = max(0, prediction)  # Ensure non-negative yield

        # Generate recommendations
        recommendations = generate_recommendations(prediction, data)

        # Determine confidence based on input quality
        confidence = "high" if data.irrigation == 1 and data.fertilizer_kg_per_ha > 80 else "moderate"

        return YieldPredictionOutput(
            predicted_yield_kg_per_ha=round(prediction, 2),
            crop_type=data.crop_type,
            confidence=confidence,
            recommendations=recommendations
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)