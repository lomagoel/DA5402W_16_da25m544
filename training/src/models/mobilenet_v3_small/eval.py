from src.models.helpers.model_eval import eval_best_model_version


MODEL_NAME = 'mobilenet_v3_small'
MLFLOW_TRACKING_URI = 'http://34.47.164.107:5000'

eval_best_model_version(MODEL_NAME, MLFLOW_TRACKING_URI)