import requests
import os
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_BASE_URL = "http://host.docker.internal:5000"
MODEL_CACHE_DIR = "model"
DEST_PATH = "/tmp/downloaded_model"

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


def download_model_local(model_name: str):
    # -----------------------------
    # Connect to MLflow
    # -----------------------------
    mlflow.set_tracking_uri(MLFLOW_BASE_URL)
    client = MlflowClient()

    # -----------------------------
    # Get model version in stage
    # -----------------------------
    model_versions = client.get_latest_versions(
        name=model_name
    )

    if not model_versions:
        raise Exception(f"No model found in stage")

    model_version = model_versions[0]
    run_id = model_version.run_id
    artifact_path = model_version.source

    print(f"Downloading model version {model_version.version}")

    # -----------------------------
    # Download artifacts
    # -----------------------------
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_path,
        dst_path=DEST_PATH
    )

    print(f"Model downloaded to: {local_path}")
    return local_path
