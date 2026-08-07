"""Airflow DAG: Image Classification Pipeline (Cloud Composer 2 / GKE).

Pipeline stages
---------------
1. download_data       — fetch Caltech-101 from Kaggle, split into folds, upload to GCS
2. preprocess_data     — validate images, check class balance, write manifest to GCS
3a. train_resnet18     — ResNet18 + Ray Tune HPO + MLflow (parallel with 3b)
3b. train_mobilenetv2  — MobileNetV2 config-driven training + MLflow (parallel with 3a)
4. pipeline_complete   — join node

All compute tasks run as Kubernetes pods (KubernetesPodOperator) using the same
Docker image built and pushed by the GitHub Actions CI pipeline.

Airflow Variables required (set via Composer UI or Terraform):
  IMAGE_URI            e.g. us-central1-docker.pkg.dev/<project>/<repo>/caltech-mobilenetv2
  IMAGE_TAG            e.g. abc1234  (updated by CI on each push)
  MLFLOW_TRACKING_URI  e.g. http://<mlflow-host>:5000
  RAY_ACCELERATOR_TYPE e.g. nvidia-tesla-t4  (leave empty to skip accelerator label)
"""
from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

from dags.dag_config import (
    GCS_DATA_PATH,
    GCS_MANIFEST_PATH,
    GPU_NODE_SELECTOR,
    GPU_RESOURCES,
    IMAGE_TEMPLATE,
    KAGGLE_SECRET_NAME,
    KSA_NAME,
    NAMESPACE,
)

# ---------------------------------------------------------------------------
# Shared env vars injected into every pod
# ---------------------------------------------------------------------------
_common_env = [
    k8s.V1EnvVar(name="GCS_DATA_PATH", value=GCS_DATA_PATH),
    k8s.V1EnvVar(
        name="MLFLOW_TRACKING_URI",
        value="{{ var.value.MLFLOW_TRACKING_URI }}",
    ),
]

# ---------------------------------------------------------------------------
# Kaggle credentials mounted from Kubernetes Secret
# ---------------------------------------------------------------------------
_kaggle_env_from = [
    k8s.V1EnvFromSource(
        secret_ref=k8s.V1SecretEnvSource(name=KAGGLE_SECRET_NAME)
    )
]

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
    download_data = KubernetesPodOperator(
        task_id="download_data",
        name="download-data",
        namespace=NAMESPACE,
        image=IMAGE_TEMPLATE,
        cmds=["python", "-m", "src.tasks.download_and_upload"],
        env_vars=[k8s.V1EnvVar(name="GCS_DATA_PATH", value=GCS_DATA_PATH)],
        env_from=_kaggle_env_from,
        service_account_name=KSA_NAME,
        is_delete_operator_pod=True,
        get_logs=True,
    )

    # ------------------------------------------------------------------
    # Task 2: Validate images and check class balance, write manifest
    # ------------------------------------------------------------------
    preprocess_data = KubernetesPodOperator(
        task_id="preprocess_data",
        name="preprocess-data",
        namespace=NAMESPACE,
        image=IMAGE_TEMPLATE,
        cmds=["python", "-m", "src.tasks.preprocess"],
        env_vars=[
            k8s.V1EnvVar(name="GCS_DATA_PATH", value=GCS_DATA_PATH),
            k8s.V1EnvVar(name="GCS_MANIFEST_PATH", value=GCS_MANIFEST_PATH),
        ],
        service_account_name=KSA_NAME,
        is_delete_operator_pod=True,
        get_logs=True,
    )

    # ------------------------------------------------------------------
    # Task 3a: Train ResNet18 with Ray Tune HPO + MLflow
    # ------------------------------------------------------------------
    train_resnet18 = KubernetesPodOperator(
        task_id="train_resnet18",
        name="train-resnet18",
        namespace=NAMESPACE,
        image=IMAGE_TEMPLATE,
        cmds=["python", "-m", "src.models.resnet18.train"],
        env_vars=_common_env + [
            k8s.V1EnvVar(
                name="RAY_ACCELERATOR_TYPE",
                value="{{ var.value.get('RAY_ACCELERATOR_TYPE', '') }}",
            ),
        ],
        container_resources=GPU_RESOURCES,
        node_selector=GPU_NODE_SELECTOR,
        service_account_name=KSA_NAME,
        is_delete_operator_pod=True,
        get_logs=True,
    )

    # ------------------------------------------------------------------
    # Task 3b: Train MobileNetV2 (config-driven) + MLflow
    # ------------------------------------------------------------------
    train_mobilenetv2 = KubernetesPodOperator(
        task_id="train_mobilenetv2",
        name="train-mobilenetv2",
        namespace=NAMESPACE,
        image=IMAGE_TEMPLATE,
        cmds=["python", "-m", "src.tasks.train_mobilenetv2"],
        env_vars=_common_env,
        container_resources=GPU_RESOURCES,
        node_selector=GPU_NODE_SELECTOR,
        service_account_name=KSA_NAME,
        is_delete_operator_pod=True,
        get_logs=True,
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
