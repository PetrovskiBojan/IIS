from fastapi.testclient import TestClient
from unittest.mock import patch
import pytest
import numpy as np
from API.app import app  # Make sure this import matches your project structure

client = TestClient(app)

@pytest.fixture
def mock_predict_return_value():
    return np.array([0.5])

def test_predict(mock_predict_return_value):
    with patch("API.app.load_model") as mock_load_model, \
         patch("API.app.load_scaler_parameters") as mock_load_scaler_parameters:
        mock_load_model.return_value.predict.return_value = mock_predict_return_value
        mock_load_scaler_parameters.return_value = (np.array([0]), np.array([1]))
        
        request_data = {
            "station_name": "1_GOSPOSVETSKA_C.___TURNERJEVA_UL._combined",
            "current_time": "2024-03-11 13:35:00",
            "temperatures_2m": [22.3, 22.7, 13.0, 13.5, 24.0, 34.5, 35.0],
            "precipitation_probabilities": [0.1, 0.05, 0.0, 0.20, 0.05, 0.3, 0.25]
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
