"""
Loads the trained model from the MLflow Model Registry.

By default this always resolves to the LATEST registered version of the
model at load/reload time — no need to know or hardcode a version number,
and no code or env-var change is needed when the training side registers
a new version. Set MLFLOW_MODEL_VERSION to pin to a specific version or
alias instead, if that's ever needed.
"""
import logging
import threading

import mlflow
import mlflow.pyfunc
import torch
import functools
torch.load = functools.partial(torch.load, map_location=torch.device("cpu"))

from app.config import settings

logger = logging.getLogger("serving_app.model_loader")

_model = None
_model_version = None
_lock = threading.Lock()


def _resolve_latest_version(client) -> str:
    """
    Queries the registry for every version of the configured model and
    returns the highest version number as a string. This is what makes
    the app pick up new training runs automatically.
    """
    versions = client.search_model_versions(f"name='{settings.MLFLOW_MODEL_NAME}'")
    if not versions:
        raise ValueError(f"No versions found for registered model '{settings.MLFLOW_MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    return latest.version


def _model_uri(resolved_version: str) -> str:
    return f"models:/{settings.MLFLOW_MODEL_NAME}/{resolved_version}"


def load_model(force: bool = False):
    """
    Loads (or reloads) the model from the MLflow registry, resolving to
    the latest version unless MLFLOW_MODEL_VERSION pins a specific one.
    Thread-safe; safe to call from /reload while requests are in flight —
    readers get the old model until the swap completes.
    """
    global _model, _model_version

    with _lock:
        if _model is not None and not force:
            return _model

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        if settings.MLFLOW_MODEL_VERSION:
            resolved_version = settings.MLFLOW_MODEL_VERSION
            logger.info(f"Using pinned version: {resolved_version}")
        else:
            resolved_version = _resolve_latest_version(client)
            logger.info(f"Auto-resolved latest version: {resolved_version}")

        uri = _model_uri(resolved_version)
        logger.info(f"Loading model '{settings.MLFLOW_MODEL_NAME}' from {uri}")

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
