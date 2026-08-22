"""
Loads the trained model from the MLflow Model Registry.

Two things your teammates' training code (train_DA25M607.py / the Ray
training jobs) needs to do for this to work:
  1. Log the model with `mlflow.pytorch.log_model(model, "model")` (or the
     appropriate flavor for whatever framework is used) inside the run.
  2. Register that model version under a fixed name, e.g.:
       mlflow.register_model(f"runs:/{run_id}/model", "image_classifier")
     and promote the version you want served to a stage (e.g. "Production")
     via the MLflow UI or `MlflowClient.transition_model_version_stage`.

This module then loads via the "models:/<name>/<stage>" URI, so a new
model version can be promoted to Production without redeploying the API —
just call POST /reload.
"""
import logging
import threading

import mlflow
import mlflow.pyfunc

from app.config import settings

logger = logging.getLogger("serving_app.model_loader")

_model = None
_model_version = None
_lock = threading.Lock()


def _model_uri() -> str:
    if settings.MLFLOW_MODEL_VERSION:
        return f"models:/{settings.MLFLOW_MODEL_NAME}/{settings.MLFLOW_MODEL_VERSION}"
    return f"models:/{settings.MLFLOW_MODEL_NAME}/{settings.MLFLOW_MODEL_STAGE}"


def load_model(force: bool = False):
    """
    Loads (or reloads) the model from the MLflow registry.
    Thread-safe; safe to call from the /reload endpoint while requests
    are in flight (readers get the old model until the swap completes).
    """
    global _model, _model_version

    with _lock:
        if _model is not None and not force:
            return _model

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        uri = _model_uri()
        logger.info(f"Loading model from {uri}")

        # Resolve the actual version number for logging/health checks
        if settings.MLFLOW_MODEL_VERSION:
            resolved_version = settings.MLFLOW_MODEL_VERSION
        else:
            try:
                versions = client.get_latest_versions(
                    settings.MLFLOW_MODEL_NAME, stages=[settings.MLFLOW_MODEL_STAGE]
                )
                resolved_version = versions[0].version if versions else "unknown"
            except Exception as e:
                logger.warning(f"Could not resolve version metadata: {e}")
                resolved_version = "unknown"

        model = mlflow.pyfunc.load_model(uri)

        _model = model
        _model_version = resolved_version
        logger.info(f"Model loaded: {settings.MLFLOW_MODEL_NAME} version={resolved_version}")

        return _model


def get_model():
    if _model is None:
        return load_model()
    return _model


def get_model_version():
    return _model_version
