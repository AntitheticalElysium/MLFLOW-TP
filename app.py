import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()

def get_mlflow_uri():
    if os.getenv("DOCKER_ENV") == "true":
        return "http://host.docker.internal:8080"
    else:
        return "http://127.0.0.1:8080"

mlflow.set_tracking_uri(uri=get_mlflow_uri())

model = None
model_version = None

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
    global model, model_version
    try:
        if version == "latest":
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{version}"
        model = mlflow.sklearn.load_model(model_uri)
        model_version = version
        return True
    except Exception as e:
        print(f"Error loading model from MLFlow: {e}")
    return False

@app.get("/")
async def root():
    return {"message": "Insurance Cost Prediction Service", "model_version": model_version}

@app.post("/predict")
async def predict(data: PredictionInput):
    global model, model_version
    if model is None:
        loaded = load_model_from_mlflow()
        if not loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/update-model")
async def update_model(update_data: ModelUpdate):
    global model, model_version
    
    loaded = load_model_from_mlflow(update_data.model_name, update_data.version)
    if loaded:
        return {"message": f"Model updated to {update_data.model_name} version {update_data.version}"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None, "model_version": model_version}
