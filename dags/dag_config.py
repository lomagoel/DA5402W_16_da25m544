"""Centralized constants for the image classification DAG.

All resource specs, GCS paths, and Kubernetes configuration live here
so the DAG file stays clean and values are easy to update.
"""
from __future__ import annotations

from kubernetes.client import models as k8s

# ---------------------------------------------------------------------------
# Docker image
# IMAGE_URI and IMAGE_TAG are set as Airflow Variables in Cloud Composer.
# The DAG template-renders them at runtime.
# ---------------------------------------------------------------------------
IMAGE_TEMPLATE = "{{ var.value.IMAGE_URI }}:{{ var.value.IMAGE_TAG }}"

# ---------------------------------------------------------------------------
# GCS paths
# ---------------------------------------------------------------------------
GCS_DATA_PATH = "gs://caltech_1000/data"
GCS_MANIFEST_PATH = "gs://caltech_1000/manifests/clean_manifest.json"

# ---------------------------------------------------------------------------
# Kubernetes / Cloud Composer 2
# ---------------------------------------------------------------------------
NAMESPACE = "composer-user-workloads"

# Kubernetes ServiceAccount that has Workload Identity bound to the GCP SA.
# Set up once: kubectl annotate serviceaccount mlops-ksa \
#   iam.gke.io/service-account=mlops-pipeline@<project>.iam.gserviceaccount.com \
#   -n composer-user-workloads
KSA_NAME = "mlops-ksa"

# GKE GPU node pool label — change if your node pool uses a different accelerator.
GPU_NODE_SELECTOR = {"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"}

GPU_RESOURCES = k8s.V1ResourceRequirements(
    limits={"nvidia.com/gpu": "1"},
    requests={"nvidia.com/gpu": "1"},
)

# Kaggle credentials are stored as a Kubernetes Secret named "kaggle-credentials"
# with keys KAGGLE_USERNAME and KAGGLE_KEY.
# Create with:
#   kubectl create secret generic kaggle-credentials \
#     --from-literal=KAGGLE_USERNAME=<user> \
#     --from-literal=KAGGLE_KEY=<key> \
#     -n composer-user-workloads
KAGGLE_SECRET_NAME = "kaggle-credentials"
