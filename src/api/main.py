import os
import json
import pickle
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from loguru import logger
import pandas as pd
import numpy as np

from src.api.schemas import CreditPredictionPayload, PredictionResponse, HealthResponse

import onnxruntime as ort

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
ONNX_PATH = os.getenv("ONNX_PATH", "model/model.onnx")
LOG_FILE = os.getenv("LOG_FILE", "production_logs.json")

# Logger Setup
logger.add(LOG_FILE, format="{message}", serialize=True, rotation="10 MB")

class ModelWrapper:
    def __init__(self, path: str, onnx_path: str = None):
        self.path = path
        self.onnx_path = onnx_path
        self.model = None
        self.session = None
        self.use_onnx = False
        self.load()

    def load(self):
        # Try loading ONNX first if available
        if self.onnx_path and os.path.exists(self.onnx_path):
            try:
                logger.info(f"Loading ONNX model from {self.onnx_path}...")
                self.session = ort.InferenceSession(self.onnx_path)
                self.input_name = self.session.get_inputs()[0].name
                self.use_onnx = True
                logger.info("ONNX model loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Failed to load ONNX model: {e}. Falling back to pickle.")

        try:
            logger.info(f"Loading model from {self.path}...")
            with open(self.path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model not found at {self.path}")
            raise RuntimeError(f"Model not found at {self.path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Error loading model: {e}")

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.model and not self.session:
            raise RuntimeError("Model not loaded")
        
        # Prepare Data
        # Ensure correct order if using generic dict. 
        # For ONNX/Pickle, we assume input_data matches expected schema.
        # Ideally we should align columns here.
        
        try:
            if self.use_onnx:
                # Convert dict values to list/array in correct order?
                # ONNX expects a specific tensor.
                # Assuming simple 1D input scaled correctly.
                # WARNING: Dictionary order is preserved in recent python, but safer to know feature list.
                # For this demo, we assume values() is sufficient or user sends compatible dict.
                
                # We convert values to float32 list
                vals = list(input_data.values())
                input_tensor = np.array([vals], dtype=np.float32)
                
                # Predict
                result = self.session.run(None, {self.input_name: input_tensor})
                # Result structure depends on converter (usually [label, probabilities])
                prediction = result[0][0]
                
                # Probabilities (map or list)
                # LightGBM ONNX usually returns a list of maps or a tensor.
                # Adjust based on observation. standard sklearn-onnx returns [label, prob_list]
                
                # Inspecting result[1] usually holds probs
                if len(result) > 1:
                    probs = result[1]
                    if isinstance(probs, list):
                        # List of dictionaries
                        probability = probs[0].get(1, 0.0) # Prob of class 1
                    else:
                        # Tensor
                        probability = probs[0][1]
                else:
                    probability = float(prediction)
                    
            else:
                # Pickle Fallback
                df = pd.DataFrame([input_data])
                prediction = self.model.predict(df)[0]
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(df)
                    probability = probs[0][1] if probs.shape[1] > 1 else probs[0][0]
                else:
                    probability = float(prediction)

            return {
                "prediction": int(prediction),
                "probability": float(probability)
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Global Model Instance
model_wrapper = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_wrapper
    model_wrapper = ModelWrapper(MODEL_PATH, ONNX_PATH)
    yield
    # Shutdown
    logger.info("Shutting down API")

app = FastAPI(title="Credit Scoring API", version="1.0.0", lifespan=lifespan)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # We log in the endpoints usually to capture input/output, 
    # but middleware catches all. 
    # For structured logging of specific prediction data, we do it in the endpoint.
    return response

@app.get("/health", response_model=HealthResponse)
def health_check():
    if not model_wrapper or (not model_wrapper.model and not model_wrapper.session):
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "ok", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: CreditPredictionPayload):
    if not model_wrapper:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    start_time = time.time()
    result = model_wrapper.predict(payload.input)
    duration = time.time() - start_time
    
    # Business Logic / Threshold
    # If the business has a specific threshold, apply it here or rely on model.predict
    # We return the raw model prediction and probability.
    
    response_data = {
        "prediction": result["prediction"],
        "probability": result["probability"],
        "threshold_used": 0.5, # Default, or fetch from config
        "features_received": len(payload.input)
    }
    
    # Log for Monitoring (Dashboard)
    log_entry = {
        "timestamp": time.time(),
        "input": payload.input,
        "prediction": response_data,
        "duration": duration,
        "status": "success"
    }
    logger.info(json.dumps(log_entry))
    
    return response_data
