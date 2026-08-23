"""
Configuration for the model serving app.
Values are read from environment variables so they can be overridden
in Docker / Kubernetes / CI without touching code.
"""
import os


class Settings:
    # MLflow
    MLFLOW_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_MODEL_NAME: str = os.environ.get("MLFLOW_MODEL_NAME", "image_classifier")
    # Stage-based loading, e.g. "Production", "Staging" (deprecated in newer MLflow).
    MLFLOW_MODEL_STAGE: str = os.environ.get("MLFLOW_MODEL_STAGE", "")
    # Version-based loading, e.g. "4". Takes priority over MLFLOW_MODEL_STAGE if set —
    # use this on MLflow instances using the new registry UI (aliases replacing stages),
    # where no stage/alias has been assigned yet.
    MLFLOW_MODEL_VERSION: str = os.environ.get("MLFLOW_MODEL_VERSION", "")

    # Inference
    DEVICE: str = os.environ.get("DEVICE", "cpu")  # "cpu" or "cuda"
    IMAGE_SIZE: int = int(os.environ.get("IMAGE_SIZE", "224"))

    # App
    APP_TITLE: str = "Image Classification Serving API"
    MAX_UPLOAD_MB: int = int(os.environ.get("MAX_UPLOAD_MB", "10"))


settings = Settings()
