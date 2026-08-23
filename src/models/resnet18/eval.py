from src.models.helpers.model_evaluate import eval_best_model_version


MODEL_NAME = 'resnet18'
MLFLOW_TRACKING_URI = 'http://34.47.164.107:5000'

eval_best_model_version(MODEL_NAME, MLFLOW_TRACKING_URI)
