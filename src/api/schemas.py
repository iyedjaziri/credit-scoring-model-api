from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class CreditPredictionPayload(BaseModel):
    """
    Schema for input data to the credit scoring model.
    Accepts a dictionary of features as expected by the model.
    """
    input: Dict[str, Any] = Field(
        ..., 
        description="Dictionary of features for the customer. Keys must match model feature names.",
        example={"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.6, "EXT_SOURCE_3": 0.4, "DAYS_BIRTH": -15000}
    )

class PredictionResponse(BaseModel):
    """
    Schema for the prediction output.
    """
    prediction: int = Field(..., description="0 for Repaid, 1 for Default")
    probability: float = Field(..., description="Probability of Default (0-1)")
    threshold_used: float = Field(..., description="Decision threshold used")
    features_received: int = Field(..., description="Number of features processed")
    
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
