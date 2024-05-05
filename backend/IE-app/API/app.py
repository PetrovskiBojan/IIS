from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
from tensorflow.keras.models import load_model
import json
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    station_name: str
    current_time: str  # Consider parsing this into a datetime object if needed
    temperatures_2m: list[float]
    precipitation_probabilities: list[float]

# MLflow setup
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

def load_scaler_parameters(station_name):
    SCALERS_DIR = os.path.join(os.path.dirname(__file__), "../data", "scaler_params")
    scaler_path = os.path.join(SCALERS_DIR, f"{station_name}_scaler_params.json")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler parameters not found for station {station_name}. Looked in {scaler_path}")
    with open(scaler_path, "r") as file:
        scaler_params = json.load(file)
    return np.array(scaler_params["min_"]), np.array(scaler_params["scale_"])

def normalize(data, min_, scale_):
    return (data - min_) / scale_

def load_model_from_mlflow(model_name):
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        raise HTTPException(status_code=404, detail=f"Production model '{model_name}' not found in MLflow.")
    model_uri = latest_version[0].source
    return mlflow.keras.load_model(model_uri)

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        model = load_model_from_mlflow(request.station_name)
        min_, scale_ = load_scaler_parameters(request.station_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    predictions = []
    for i in range(7):  # Generate predictions for the next 7 hours
        temperatures = np.array(request.temperatures_2m[i:i+1]).reshape(-1, 1)
        precipitations = np.array(request.precipitation_probabilities[i:i+1]).reshape(-1, 1)
        input_features = np.hstack((temperatures, precipitations))
        normalized_features = normalize(input_features, min_, scale_).reshape(1, -1, 2)

        prediction = model.predict(normalized_features).flatten()[0]
        predictions.append(int(round(prediction)))

    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
