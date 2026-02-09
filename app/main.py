from fastapi import FastAPI, Query
from app.schemas import PredictionRequest, PredictionResponse
from app.model import predict_digit
import numpy as np

app = FastAPI(title="MNIST MLflow Registry API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    model_name: str = Query(..., example="mnist_model"),
    stage: str = Query("Production", example="Production")
):
    image = np.array(request.image)

    label, probs = predict_digit(model_name, stage, image)

    return {
        "model_name": model_name,
        "stage": stage,
        "predicted_label": label,
        "probabilities": probs
    }
