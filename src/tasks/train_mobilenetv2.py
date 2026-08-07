"""Task wrapper: train MobileNetV2 on Caltech-101 using config from training_config.yaml.

Reads from environment (both override YAML values if set):
  GCS_DATA_PATH        - path to dataset root (e.g. /gcs/caltech_1000/data or a local path)
  MLFLOW_TRACKING_URI  - MLflow tracking server URI
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from src.train import train

CONFIG_PATH = Path(__file__).resolve().parent.parent / "training_config.yaml"


def main() -> None:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    gcs_data_path = os.environ.get("GCS_DATA_PATH")
    if gcs_data_path:
        config["data_dir"] = gcs_data_path

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        config["tracking_uri"] = mlflow_uri

    print(f"Training MobileNetV2 | data_dir={config['data_dir']} | tracking_uri={config.get('tracking_uri')}")
    train(config)


if __name__ == "__main__":
    main()
