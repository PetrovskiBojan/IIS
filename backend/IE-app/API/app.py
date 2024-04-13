from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
from tensorflow.keras.models import load_model
import json
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware

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

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
SCALERS_DIR = os.path.join(os.path.dirname(__file__), "../data", "scaler_params")

def load_scaler_parameters(station_name):
    scaler_path = os.path.join(SCALERS_DIR, f"{station_name}_scaler_params.json")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler parameters not found for station {station_name}. Looked in {scaler_path}")
    with open(scaler_path, "r") as file:
        scaler_params = json.load(file)
    return np.array(scaler_params["min_"]), np.array(scaler_params["scale_"])

def normalize(data, min_, scale_):
    return (data - min_) / scale_

@app.post("/predict")
async def predict(request: PredictionRequest):
    model_path = os.path.join(MODELS_DIR, f"{request.station_name}.h5")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found for station {request.station_name}. Looked in {model_path}")

    try:
        min_, scale_ = load_scaler_parameters(request.station_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    model = load_model(model_path)

    predictions = []
    for i in range(7):  # Generate predictions for the next 7 hours
        temperatures = np.array(request.temperatures_2m[i:i+1]).reshape(-1, 1)
        precipitations = np.array(request.precipitation_probabilities[i:i+1]).reshape(-1, 1)
        input_features = np.hstack((temperatures, precipitations))
        normalized_features = normalize(input_features, min_, scale_).reshape(1, -1, 2)  # Reshape for the model

        prediction = model.predict(normalized_features).flatten()[0]
        predictions.append(int(round(prediction)))

    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
