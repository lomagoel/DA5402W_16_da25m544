import mlflow

from src.models.helpers.model_trainer import ModelTrainer
from src.models.helpers.ray_helper import RayDriver


class ModelPipeline():

    def __init__(self, mlflow_experiment_name, mlflow_tracking_uri):
        print('Connecting to MLFlow', mlflow_tracking_uri)
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        self.mlflow_config = {
            'tracking_uri': mlflow_tracking_uri,
            'experiment_name': mlflow_experiment_name,
        }

    def tune(self, model, dataset, num_trials, search_space):
        print('Tuning...')
        tuner = RayDriver(model, {'cpu': 1, 
                                  'gpu':0,# 1/num_trials, 
                                  'accelerator_type:RTX': 0},#1/num_trials}
                                   self.mlflow_config)
        best_config = tuner.tune(dataset, search_space, num_samples=num_trials)
        tuner.shutdown()
        return best_config

    def train(self, model, train_dataset, test_dataset, config, mlflow_artifact_name):
        print('Best config model training...')
        with mlflow.start_run(
            run_name='BEST_MODEL'
        ):
            model_engine = ModelTrainer(model, config['batch_size'], config['optim'])
            model_engine.train(train_dataset)
            test_metrics = model_engine.evaluate(test_dataset)
            mlflow.log_metrics(test_metrics)
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name=mlflow_artifact_name,
                serialization_format='pickle'
            )

