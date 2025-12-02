"""
FastAPI application for ML model training and prediction.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from preprocessing import FeaturePreprocessor
from models import ModelTrainer, XGBOOST_AVAILABLE

app = FastAPI(title="Immo Eliza ML API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use a database)
PREPROCESSOR = None
TRAINER = None
DATA_CACHE = {}


class ModelParams(BaseModel):
    """Model parameters for training."""
    Linear_Regression: Dict[str, Any] = Field(default_factory=dict)
    SVR: Dict[str, Any] = Field(default_factory=dict)
    Decision_Tree: Dict[str, Any] = Field(default_factory=dict)
    Random_Forest: Dict[str, Any] = Field(default_factory=dict)
    XGBoost: Dict[str, Any] = Field(default_factory=dict)


class TrainingRequest(BaseModel):
    """Request for model training."""
    csv_path: str
    model_params: Optional[Dict[str, Dict[str, Any]]] = None
    test_size: float = 0.2
    cv_folds: int = 5
    use_capping: bool = True
    capping_percentiles: List[float] = [1, 99]


class PredictionRequest(BaseModel):
    """Request for prediction."""
    model_name: str
    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    """Response for prediction."""
    prediction: float
    model_name: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Immo Eliza ML API",
        "version": "1.0.0",
        "xgboost_available": XGBOOST_AVAILABLE,
        "endpoints": {
            "/train": "Train models",
            "/predict": "Make predictions",
            "/models": "Get available models",
            "/results": "Get training results",
            "/overfitting": "Check overfitting status"
        }
    }


@app.post("/train")
async def train_models(request: TrainingRequest):
    """
    Train all models with given parameters.

    Args:
        request: Training configuration

    Returns:
        Training results for all models
    """
    global PREPROCESSOR, TRAINER, DATA_CACHE

    try:
        # Load data
        print(f"Loading data from {request.csv_path}")
        df = pd.read_csv(request.csv_path)
        DATA_CACHE["raw_data"] = df

        # Initialize preprocessor
        PREPROCESSOR = FeaturePreprocessor(
            use_capping=request.use_capping,
            capping_percentiles=tuple(request.capping_percentiles)
        )

        # Fit and transform
        print("Preprocessing data...")
        X, y = PREPROCESSOR.fit_transform(df)

        # Initialize trainer
        model_params = request.model_params or ModelTrainer.DEFAULT_PARAMS
        # Convert underscores to spaces in keys
        model_params = {k.replace("_", " "): v for k, v in model_params.items()}

        TRAINER = ModelTrainer(model_params=model_params)

        # Split data
        TRAINER.train_test_split_data(X, y, test_size=request.test_size)

        # Train all models
        print("Training models...")
        results = TRAINER.train_all_models(cv_folds=request.cv_folds)

        # Save models and preprocessor
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        PREPROCESSOR.save(model_dir / "preprocessor.json")
        TRAINER.save(model_dir / "trainer.json")

        return {
            "status": "success",
            "message": "Models trained successfully",
            "results": results,
            "data_shape": {
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "train_size": int(len(TRAINER.y_train)),
                "test_size": int(len(TRAINER.y_test))
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction with a specific model.

    Args:
        request: Prediction request with features

    Returns:
        Prediction result
    """
    global PREPROCESSOR, TRAINER

    if PREPROCESSOR is None or TRAINER is None:
        raise HTTPException(
            status_code=400,
            detail="Models not trained yet. Please train models first."
        )

    try:
        # Convert features to DataFrame
        df = pd.DataFrame([request.features])

        # Preprocess
        X = PREPROCESSOR.transform(df)

        # Predict
        y_pred = TRAINER.predict(request.model_name, X)

        # Inverse transform
        price = PREPROCESSOR.inverse_transform_target(y_pred)[0]

        return PredictionResponse(
            prediction=float(price),
            model_name=request.model_name
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def get_models():
    """Get list of available models and their parameters."""
    if TRAINER is None:
        return {
            "status": "not_trained",
            "available_models": list(ModelTrainer.DEFAULT_PARAMS.keys()),
            "default_params": ModelTrainer.DEFAULT_PARAMS
        }

    return {
        "status": "trained",
        "available_models": list(TRAINER.models.keys()),
        "parameters": TRAINER.model_params
    }


@app.get("/results")
async def get_results():
    """Get training results for all models."""
    if TRAINER is None or not TRAINER.results:
        raise HTTPException(
            status_code=400,
            detail="No training results available. Train models first."
        )

    return {
        "results": TRAINER.results,
        "best_model": TRAINER.get_best_model(metric="test_r2")
    }


@app.get("/overfitting")
async def check_overfitting(threshold: float = 0.1):
    """
    Check overfitting status for all models.

    Args:
        threshold: Maximum acceptable R² gap between train and test

    Returns:
        Overfitting status for each model
    """
    if TRAINER is None or not TRAINER.results:
        raise HTTPException(
            status_code=400,
            detail="No training results available. Train models first."
        )

    overfitting_status = TRAINER.detect_overfitting(threshold=threshold)

    return {
        "threshold": threshold,
        "overfitting_status": overfitting_status
    }


@app.get("/feature-info")
async def get_feature_info():
    """Get information about available features."""
    return {
        "numeric_features": list(FeaturePreprocessor.NUMERIC),
        "categorical_features": list(FeaturePreprocessor.CATEGORICAL),
        "binary_features": list(FeaturePreprocessor.BINARY),
        "all_features": list(FeaturePreprocessor.ALL_FEATURES)
    }


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """Upload CSV data for training."""
    try:
        # Save uploaded file
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        file_path = data_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Load and validate
        df = pd.read_csv(file_path)

        return {
            "status": "success",
            "message": f"File uploaded successfully",
            "file_path": str(file_path),
            "shape": df.shape,
            "columns": list(df.columns)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
