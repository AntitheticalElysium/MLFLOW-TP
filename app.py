import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import random
from contextlib import asynccontextmanager

# Global variables
current_model = None
current_model_version = None
next_model = None
next_model_version = None
canary_p = 0.8

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global current_model, current_model_version, next_model, next_model_version
    model, version = load_model_from_mlflow()
    current_model = model
    current_model_version = version
    next_model = model
    next_model_version = version
    yield
    # Shutdown (nothing to clean up for now)

app = FastAPI(lifespan=lifespan)

def get_mlflow_uri():
    if os.getenv("DOCKER_ENV") == "true":
        return "http://host.docker.internal:8080"
    else:
        return "http://127.0.0.1:8080"

mlflow.set_tracking_uri(uri=get_mlflow_uri())

class PredictionInput(BaseModel):
    age: float
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

class ModelUpdate(BaseModel):
    model_name: str
    version: str

def preprocess_input(data: PredictionInput):
    sex_enc = 1 if data.sex.lower() == 'male' else 0
    smoker_enc = 1 if data.smoker.lower() == 'yes' else 0
    
    region_mapping = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
    region_enc = region_mapping.get(data.region.lower(), 0)
    
    feature_names = ['age', 'sex_enc', 'bmi', 'children', 'smoker_enc', 'region_enc']
    feature_values = [data.age, sex_enc, data.bmi, data.children, smoker_enc, region_enc]
    
    df = pd.DataFrame([feature_values], columns=feature_names)
    return df

def load_model_from_mlflow(model_name="random_forest_model", version="latest"):
    try:
        if version == "latest":
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{version}"
        model = mlflow.sklearn.load_model(model_uri)
        return model, version
    except Exception as e:
        print(f"Error loading model from MLFlow: {e}")
    return None, None

@app.get("/")
async def root():
    return {"message": "Insurance Cost Prediction Service", "model_version": current_model_version}

@app.post("/predict")
async def predict(data: PredictionInput):
    global current_model, next_model, canary_p
    if current_model is None or next_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    use_current = random.random() < canary_p
    model_used = "current" if use_current else "next"
    model = current_model if use_current else next_model
    try:
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)[0]
        return {"prediction": float(prediction), "model_used": model_used}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/update-model")
async def update_model(update_data: ModelUpdate):
    global next_model, next_model_version
    model, version = load_model_from_mlflow(update_data.model_name, update_data.version)
    if model:
        next_model = model
        next_model_version = version
        return {"message": f"Next model updated to {update_data.model_name} version {update_data.version}"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@app.post("/accept-next-model")
async def accept_next_model():
    global current_model, current_model_version, next_model, next_model_version
    # Set current model to next model
    current_model = next_model
    current_model_version = next_model_version
    # Keep next model the same (both current and next are now the same)
    return {"message": "Next model accepted as current", "current_version": current_model_version}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "current_model_loaded": current_model is not None,
        "next_model_loaded": next_model is not None,
        "current_model_version": current_model_version,
        "next_model_version": next_model_version
    }
