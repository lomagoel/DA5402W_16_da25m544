import mlflow
from mlflow.tracking import MlflowClient

from src.models.helpers.model_trainer import ModelTrainer
from src.models.helpers.ray_helper import RayDriver


class ModelPipeline():

    def __init__(self, mlflow_experiment_name, mlflow_tracking_uri):
        print('Connecting to MLFlow server:', mlflow_tracking_uri)
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        self.mlflow_config = {
            'tracking_uri': mlflow_tracking_uri,
            'experiment_name': mlflow_experiment_name,
        }

    def get_model(self, model_name):
        print(f'Fetching model {model_name}...')
        mlflow_client = MlflowClient()
        trained_model = mlflow_client.search_registered_models(filter_string=f"name='{model_name}'")

        if trained_model:
            model_uri = f"models:/{model_name}/latest" 
            model = mlflow.pytorch.load_model(model_uri)
            return model

        return None

    def tune(self, model, val_dataset, num_trials, search_space):
        print('Tuning...')
        tuner = RayDriver(model, {'cpu': 1, 'gpu': 1/num_trials, 'accelerator_type:RTX': 1/num_trials}, self.mlflow_config)
        best_config = tuner.tune(val_dataset.dataset, search_space, num_samples=num_trials)
        tuner.shutdown()
        return best_config

    def train(self, model, train_dataset, test_dataset, config, model_name):
        print('Best config model training...')
        with mlflow.start_run(
            run_name=f'best_model_{model_name}'
        ):
            model_engine = ModelTrainer(model, config['batch_size'], config['optim'])
            model_engine.train(train_dataset.dataset)
            test_metrics = model_engine.evaluate(test_dataset.dataset)
            mlflow.log_metrics(test_metrics)
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name=f'{model_name}_models',
                serialization_format='pickle',
                registered_model_name=model_name
            )

