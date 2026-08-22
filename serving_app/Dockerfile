# --- Base image ---
FROM python:3.10-slim

WORKDIR /app

# System deps for Pillow / mlflow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: if the registered model uses the PyTorch flavor (mlflow.pytorch.log_model),
# uncomment the line below — pyfunc loading needs the underlying framework installed.
# For TensorFlow models, install tensorflow instead.
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY app ./app

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# MLFLOW_TRACKING_URI / MLFLOW_MODEL_NAME / MLFLOW_MODEL_STAGE are passed
# at runtime via `docker run -e ...` or docker-compose / k8s env vars.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
