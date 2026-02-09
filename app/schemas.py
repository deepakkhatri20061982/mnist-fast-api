from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    image: List[List[float]]  # 28x28

class PredictionResponse(BaseModel):
    model_name: str
    stage: str
    predicted_label: int
    probabilities: List[float]

