import mlflow

from src.models.helpers.data_helper import train_val_split
from src.models.helpers.model_trainer import ModelTrainer
from src.models.helpers.ray_helper import RayDriver


class ModelPipeline():

    def __init__(self, mlflow_experiment_name, mlflow_tracking_uri, num_classes):
        print('Connecting to MLFlow server:', mlflow_tracking_uri)
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        try:
            mlflow.create_experiment(name=mlflow_experiment_name)
        except Exception:
            mlflow.set_experiment(mlflow_experiment_name)

        self.mlflow_config = {
            'experiment_name': mlflow_experiment_name,
            'tracking_uri': mlflow_tracking_uri,
        }
        self.num_classes = num_classes

    def tune(self, model, val_dataset, num_trials, search_space):
        print('Tuning...')
        val_tune_dataset, val_test_dataset = train_val_split(val_dataset, test_size=0.2, apply_stratify=False)
        tuner = RayDriver(model, {'cpu': 1, 'gpu': 1/num_trials, 'accelerator_type:RTX': 1/num_trials}, self.mlflow_config, self.num_classes)
        best_config = tuner.tune((val_tune_dataset, val_test_dataset), search_space, num_samples=num_trials)
        tuner.shutdown()
        return best_config

    def train(self, model, train_dataset, val_dataset, test_dataset, config, epochs, model_name):
        print('Best config model training...')
        with mlflow.start_run(
            run_name=f'best_model_{model_name}'
        ):
            model_engine = ModelTrainer(model, config['batch_size'], config['optim'], self.num_classes)
            model_engine.train(train_dataset, val_dataset, epochs)
            test_metrics = model_engine.evaluate(test_dataset)
            mlflow.log_metrics(test_metrics)
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name=f'{model_name}_models',
                serialization_format='pickle',
                registered_model_name=model_name
            )
