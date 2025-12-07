from fastapi.testclient import TestClient
from src.api.main import app
import pytest


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

def test_prediction_endpoint():
    # Mock payload
    payload = {
        "input": {
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.5,
            "EXT_SOURCE_3": 0.5,
            "DAYS_BIRTH": -15000,
            # Add minimal features required by your model
        }
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code in [200, 400] 
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
