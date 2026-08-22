from src.models.helpers.model_evaluate import eval_best_model_version


MODEL_NAME = 'mobilenet_v3_small'
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'

eval_best_model_version(MODEL_NAME, MLFLOW_TRACKING_URI)
