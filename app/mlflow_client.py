import requests
import os

MLFLOW_BASE_URL = "http://host.docker.internal:5000"
MODEL_CACHE_DIR = "model"

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def get_run_id_from_model_name(model_name: str, stage: str = "Production"):
    """
    Resolve run_id from MLflow Model Registry using model name + stage
    """
    url = f"{MLFLOW_BASE_URL}/api/2.0/mlflow/registered-models/get-latest-versions"
    params = {"name": model_name}

    response = requests.get(url, params=params)
    response.raise_for_status()

    versions = response.json()["model_versions"]
    return versions[0]["run_id"], versions[0]["source"]

def download_model(run_id: str, source: str, artifact_path: str = "model/model.pkl"):
    local_model_path = os.path.join(MODEL_CACHE_DIR, f"{run_id}.pkl")

    if os.path.exists(local_model_path):
        return local_model_path

    url = f"{MLFLOW_BASE_URL}/get-artifact"
    params = {
        "run_uuid": run_id,
        "path": artifact_path
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    with open(local_model_path, "wb") as f:
        f.write(response.content)

    return local_model_path
