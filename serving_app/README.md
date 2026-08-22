# Image Classification Serving API

FastAPI service that loads a trained model from the MLflow Model Registry
and serves predictions over HTTP. This is the deployment piece of the
MTech MLOps capstone project (owned by Pruthwiraj) — it sits on top of
Varun's training pipeline (`image_classification_pipeline`, Ray + MLflow).

## How it fits together

```
[Training pipeline: Ray Tune + MLflow]  (Varun)
        │  logs + registers model
        ▼
[MLflow Model Registry]  ──stage: "Production"──▶  [This serving app]
                                                            │
                                                    POST /predict
                                                    POST /predict/base64
                                                            │
                                                            ▼
                                                    {predicted_class, confidence}
```

## Endpoints

| Method | Path              | Description                                      |
|--------|-------------------|---------------------------------------------------|
| GET    | `/`               | Browser UI — drag/drop or choose an image, see prediction |
| GET    | `/health`         | Model load status + version                       |
| POST   | `/predict`        | Multipart file upload → prediction                 |
| POST   | `/predict/base64` | JSON `{"image_base64": "..."}` → prediction        |
| POST   | `/reload`         | Force-reload model from registry (after promotion) |

The `/` page is a lightweight HTML+JS frontend (`app/templates/index.html`,
`app/static/style.css`) that calls `/predict` directly — no separate
frontend server needed. It shows model health, lets you drag/drop or pick
an image, and displays the predicted class plus a top-k confidence table.
Useful for manual testing and demos (including the required 10-minute
demo video — this gives you a visual "API deployment" segment for free).

## Prerequisites — what the training side needs to do

For this app to actually load a model, the training pipeline needs to:

1. Log the model inside the MLflow run:
   ```python
   import mlflow.pytorch  # or mlflow.tensorflow, depending on framework
   mlflow.pytorch.log_model(model, "model")
   ```
2. Register it under a fixed name and promote a version to a stage:
   ```python
   run_id = mlflow.active_run().info.run_id
   result = mlflow.register_model(f"runs:/{run_id}/model", "image_classifier")

   from mlflow.tracking import MlflowClient
   MlflowClient().transition_model_version_stage(
       name="image_classifier", version=result.version, stage="Production"
   )
   ```
3. Ideally also log a `labels.json` artifact with the class names, so
   `app/inference.py`'s placeholder `CLASS_NAMES` list can be replaced with
   the real ones (currently a `class_0..class_100` placeholder for Caltech-101).

## Run locally (without Docker)

```bash
pip install -r requirements.txt
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_MODEL_NAME=image_classifier
export MLFLOW_MODEL_STAGE=Production
uvicorn app.main:app --reload --port 8000
```

## Run with Docker Compose (includes a local MLflow server for testing)

```bash
docker compose up --build
```

Then:
```bash
curl -X POST -F "file=@some_image.jpg" http://localhost:8000/predict
```

In the real team setup, don't run the bundled `mlflow` service — instead
point `MLFLOW_TRACKING_URI` at wherever Varun's training pipeline already
logs runs, and drop the `mlflow` service from `docker-compose.yml`.

## Config (env vars)

| Variable              | Default                  | Meaning                                  |
|------------------------|---------------------------|-------------------------------------------|
| `MLFLOW_TRACKING_URI`  | `http://localhost:5000`   | MLflow server URL                        |
| `MLFLOW_MODEL_NAME`    | `image_classifier`        | Registered model name                    |
| `MLFLOW_MODEL_STAGE`   | `Production`               | Stage to load (`Production`/`Staging`/version) |
| `DEVICE`               | `cpu`                       | `cpu` or `cuda`                          |
| `IMAGE_SIZE`           | `224`                        | Resize dimension before inference        |
| `MAX_UPLOAD_MB`        | `10`                         | Max upload size for `/predict`           |

## Things still to wire up (TODOs in code)

- `app/inference.py`: `CLASS_NAMES` placeholder, and the preprocessing
  (resize/normalize) must match whatever transform the training code used —
  check `preprocessing/augment.py` in the training repo and align them.
- `Dockerfile`: uncomment the `torch` (or `tensorflow`) install line
  depending on which framework the registered model actually uses —
  `mlflow.pyfunc.load_model` needs the matching framework installed.
- Monitoring hooks: the latency-logging middleware in `main.py` is a
  starting point for the "API latency / prediction logging" requirement —
  extend with a Prometheus client + `/metrics` endpoint, or ship logs to
  Kibana (Amol's piece), depending on what the team settles on.
- CI/CD: this repo is a natural target for the GitHub Actions/Jenkins
  pipeline — e.g. run `pytest`, build the Docker image, push to a registry.
