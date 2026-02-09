import joblib
import numpy as np
from app.mlflow_client import (
    get_run_id_from_model_name,
    download_model
)

_loaded_models = {}

def load_model(model_name: str, stage: str):
    key = f"{model_name}:{stage}"

    if key not in _loaded_models:
        run_id, source = get_run_id_from_model_name(model_name, stage)
        model_path = download_model(run_id=run_id, source = source)
        _loaded_models[key] = joblib.load(model_path)

    return _loaded_models[key]

def predict_digit(model_name: str, stage: str, image: np.ndarray):
    model = load_model(model_name, stage)

    image = image.astype("float32")
    if image.max() > 1.0:
        image /= 255.0

    image = image.reshape(1, -1)

    probabilities = model.predict_proba(image)[0]
    predicted_label = int(np.argmax(probabilities))

    return predicted_label, probabilities.tolist()
