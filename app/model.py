import os
import joblib
import numpy as np
from app.mlflow_client import (
    get_run_id_from_model_name,
    download_model,
    download_model_local
)

_loaded_models = {}

def load_model(model_name: str, stage: str):
    key = f"{model_name}:{stage}"

    if stage.lower() == "local":
        # Load from project directory
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "model", "logreg_model.pkl")
        _loaded_models[key] = joblib.load(model_path)
    else:
        if key not in _loaded_models:
            # run_id, source = get_run_id_from_model_name(model_name, stage)
            # model_path = download_model(run_id=run_id, source = source)
            model_path = download_model_local(model_name)
            _loaded_models[key] = joblib.load(model_path + "/model.pkl")

    return _loaded_models[key]

def predict_digit(model_name: str, stage: str, image: np.ndarray):
    model = load_model(model_name, stage)

    image = image.astype("float32")
    if image.max() > 1.0:
        image /= 255.0

    image = image.reshape(1, -1)

    probabilities = model.predict_proba(image)[0]
    predicted_label = int(np.argmax(probabilities))

    return predicted_label, probabilities
