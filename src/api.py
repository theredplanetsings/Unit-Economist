import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import joblib
import logging
from pathlib import Path

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from feature_engineering import FeatureEngineer
    from model_trainer import ModelTrainer
except ImportError:
    from src.feature_engineering import FeatureEngineer
    from src.model_trainer import ModelTrainer

app = FastAPI(title="Apartment Rent Prediction API", version="1.0.0")

# Global variables for model and feature engineer
model = None
feature_engineer = None
model_name = None

class ApartmentFeatures(BaseModel):
    """Input features for apartment rent prediction."""
    bedrooms: int
    bathrooms: float
    square_feet: float
    floor: int
    building_age: int
    latitude: float
    longitude: float
    has_elevator: bool
    has_doorman: bool
    has_gym: bool
    has_parking: bool
    has_laundry: bool
    has_pool: bool
    pet_friendly: bool
    neighbourhood: str
    neighbourhood_crime_rate: float
    distance_to_subway: float
    distance_to_downtown: float
    nearby_restaurants: int
    nearby_schools_rating: float
    building_type: str = "apartment"
    heating_type: str = "central"

class PredictionResponse(BaseModel):
    """Response model for rent prediction."""
    predicted_rent: float
    confidence_interval: Optional[Dict[str, float]] = None
    model_used: str
    features_used: List[str]

@app.on_event("startup")
async def load_model():
    """Load the trained model and feature engineer on startup."""
    global model, feature_engineer, model_name
    
    try:
        # Try to load the fitted feature engineer
        feature_engineer_path = Path("models") / "feature_engineer.pkl"
        if feature_engineer_path.exists():
            feature_engineer = joblib.load(feature_engineer_path)
            logging.info("Loaded fitted feature engineer")
        else:
            # Initialize fresh feature engineer if no saved one exists
            feature_engineer = FeatureEngineer()
            logging.warning("No saved feature engineer found. Using fresh instance.")
        
        # Try to load the best model (xgboost was the best performing)
        model_name = "xgboost"  # Use the best model from training
        model_path = Path("models") / f"{model_name}.pkl"
        
        if model_path.exists():
            model = joblib.load(model_path)
            logging.info(f"Loaded model: {model_name}")
        else:
            # Fallback to random forest
            model_name = "random_forest"
            model_path = Path("models") / f"{model_name}.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                logging.info(f"Loaded fallback model: {model_name}")
            else:
                logging.warning("No pre-trained model found. Please train a model first.")
            
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Apartment Rent Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_name if model else None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_rent(apartment: ApartmentFeatures):
    """Predict rent for a single apartment."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([apartment.dict()])
        
        # Prepare features using the same pipeline
        X_processed, _ = feature_engineer.prepare_features(input_data, fit_scalers=False)
        
        # Make prediction
        predicted_rent = model.predict(X_processed)[0]
        
        # Calculate confidence interval (rough estimate for tree-based models)
        confidence_interval = None
        if hasattr(model, 'estimators_'):  # For ensemble methods
            predictions = [estimator.predict(X_processed)[0] for estimator in model.estimators_]
            std_pred = np.std(predictions)
            confidence_interval = {
                "lower": float(predicted_rent - 1.96 * std_pred),
                "upper": float(predicted_rent + 1.96 * std_pred)
            }
        
        return PredictionResponse(
            predicted_rent=float(predicted_rent),
            confidence_interval=confidence_interval,
            model_used=model_name,
            features_used=feature_engineer.get_feature_importance_names()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
async def batch_predict_rent(apartments: List[ApartmentFeatures]):
    """Predict rent for multiple apartments."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    try:
        # Convert inputs to DataFrame
        input_data = pd.DataFrame([apt.dict() for apt in apartments])
        
        # Prepare features
        X_processed, _ = feature_engineer.prepare_features(input_data, fit_scalers=False)
        
        # Make predictions
        predictions = model.predict(X_processed)
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "apartment_index": i,
                "predicted_rent": float(pred),
                "input_features": apartments[i].dict()
            })
        
        return {
            "predictions": results,
            "model_used": model_name,
            "total_predictions": len(predictions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    model_info = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "features_count": len(feature_engineer.get_feature_importance_names()) if feature_engineer else 0,
        "feature_names": feature_engineer.get_feature_importance_names() if feature_engineer else []
    }
    
    # Add model-specific info
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_engineer.get_feature_importance_names(),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        model_info["feature_importance"] = importance_df.head(10).to_dict('records')
    
    return model_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
