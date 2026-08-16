"""Airflow DAG: Image Classification Pipeline.

Pipeline stages
---------------
1. download_data       — fetch Caltech-101 from Kaggle, split into folds, upload to GCS
2. preprocess_data     — validate images, check class balance, write manifest to GCS
3a. train_resnet18     — ResNet18 + Ray Tune HPO + MLflow (parallel with 3b)
3b. train_mobilenetv2  — MobileNetV2 config-driven training + MLflow (parallel with 3a)
4. pipeline_complete   — join node

Airflow Variables required (set via Composer UI or Terraform):
  MLFLOW_TRACKING_URI  e.g. http://<mlflow-host>:5000
  RAY_ACCELERATOR_TYPE e.g. nvidia-tesla-t4  (leave empty to skip accelerator label)
  KAGGLE_USERNAME      Kaggle account username
  KAGGLE_KEY           Kaggle API key
"""
from __future__ import annotations

import os
import subprocess

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

from dags.dag_config import (
    GCS_DATA_PATH,
    GCS_MANIFEST_PATH,
)


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def _run_download_data() -> None:
    os.environ["GCS_DATA_PATH"] = GCS_DATA_PATH
    os.environ["KAGGLE_USERNAME"] = Variable.get("KAGGLE_USERNAME", default_var="")
    os.environ["KAGGLE_KEY"] = Variable.get("KAGGLE_KEY", default_var="")
    from src.tasks.download_and_upload import main
    main()


def _run_preprocess_data() -> None:
    os.environ["GCS_DATA_PATH"] = GCS_DATA_PATH
    os.environ["GCS_MANIFEST_PATH"] = GCS_MANIFEST_PATH
    from src.tasks.preprocess import main
    main()


def _run_train_resnet18() -> None:
    # resnet18/train.py executes training at module-import level, so run it
    # as a subprocess to avoid side effects at DAG parse time.
    env = {
        **os.environ,
        "GCS_DATA_PATH": GCS_DATA_PATH,
        "MLFLOW_TRACKING_URI": Variable.get("MLFLOW_TRACKING_URI", default_var=""),
        "RAY_ACCELERATOR_TYPE": Variable.get("RAY_ACCELERATOR_TYPE", default_var=""),
    }
    subprocess.run(
        ["python", "-m", "src.models.resnet18.train"],
        env=env,
        check=True,
    )


def _run_train_mobilenetv2() -> None:
    os.environ["GCS_DATA_PATH"] = GCS_DATA_PATH
    os.environ["MLFLOW_TRACKING_URI"] = Variable.get("MLFLOW_TRACKING_URI", default_var="")
    from src.tasks.train_mobilenetv2 import main
    main()


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="image_classification_pipeline",
    description="End-to-end Caltech-101 training: download → preprocess → ResNet18 & MobileNetV2",
    schedule="@weekly",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops", "caltech101", "pytorch"],
) as dag:

    # ------------------------------------------------------------------
    # Task 1: Download Caltech-101 from Kaggle, split, upload to GCS
    # ------------------------------------------------------------------
    download_data = PythonOperator(
        task_id="download_data",
        python_callable=_run_download_data,
    )

    # ------------------------------------------------------------------
    # Task 2: Validate images and check class balance, write manifest
    # ------------------------------------------------------------------
    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=_run_preprocess_data,
    )

    # ------------------------------------------------------------------
    # Task 3a: Train ResNet18 with Ray Tune HPO + MLflow
    # ------------------------------------------------------------------
    train_resnet18 = PythonOperator(
        task_id="train_resnet18",
        python_callable=_run_train_resnet18,
    )

    # ------------------------------------------------------------------
    # Task 3b: Train MobileNetV2 (config-driven) + MLflow
    # ------------------------------------------------------------------
    train_mobilenetv2 = PythonOperator(
        task_id="train_mobilenetv2",
        python_callable=_run_train_mobilenetv2,
    )

    # ------------------------------------------------------------------
    # Task 4: Join — completes when both training branches finish
    # ------------------------------------------------------------------
    pipeline_complete = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule="all_done",
    )

    # ------------------------------------------------------------------
    # Dependencies
    # ------------------------------------------------------------------
    download_data >> preprocess_data >> [train_resnet18, train_mobilenetv2] >> pipeline_complete
    # Yet to decide when to download the data, onetime or every time
# TODO Add validation step
# TODO Add deploy step
# TODO develop scripts for validation and deploy - Varun/Amol
# TODO Add airflow steps for above
# TODO Explore to add an option to run serial vs parallel
# TODO Explore if the mobilenetv2 can be shown as baseline model and resnet18 bringing in better results
# TODO Retrain the already deployed model when new data arrivers after deciding on one of the models based on initial training.
