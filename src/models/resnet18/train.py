import torch.nn as nn
from torchvision import datasets, models
from data_helper import train_val_split
import mlflow
from helpers.model_trainer import ModelTrainer
from helpers.ray_helper import RayDriver, uniform_space, random_choice


MLFLOW_EXPERIMENT_NAME = 'MLOPS_PROJECT'
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
NUM_TUNE_TRIALS = 5


caltech101_data = datasets.Caltech101(root='/tmp/data', download=True)

train_dataset, test_dataset = train_val_split(caltech101_data)
_, val_dataset = train_val_split(train_dataset.dataset)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_final_layer = model.fc.in_features
model.fc = nn.Linear(in_final_layer, len(caltech101_data.categories))


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

mlflow_config = {
    'tracking_uri': MLFLOW_TRACKING_URI,
    'experiment_name': MLFLOW_EXPERIMENT_NAME,
}


search_space = {
    'optim':{
        'lr': uniform_space(1e-4, 1e-2),
        'weight_decay': uniform_space(1e-6, 1e-3)
    },
    'batch_size': random_choice([32, 64]),
}

tuner = RayDriver(model, {'cpu': 1, 'gpu': 1/NUM_TUNE_TRIALS, 'accelerator_type:RTX': 1/NUM_TUNE_TRIALS}, mlflow_config)
best_config = tuner.tune(val_dataset.dataset, search_space, num_samples=5)

tuner.shutdown()


with mlflow.start_run(
    run_name='BEST_MODEL'
):

    model_engine = ModelTrainer(model, best_config['batch_size'], best_config['optim'])
    model_engine.train(train_dataset.dataset)
    test_metrics = model_engine.evaluate(test_dataset.dataset)
    mlflow.log_metrics(test_metrics)
    mlflow.pytorch.log_model(
        pytorch_model=model,
        name='resnet18',
        serialization_format='pickle'
    )


