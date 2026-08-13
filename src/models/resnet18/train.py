import torch.nn as nn
from torchvision import datasets, models

from src.models.helpers.model_pipeline import ModelPipeline
from src.models.helpers.data_helper import train_val_split
from src.models.helpers.ray_helper import uniform_space, random_choice


MLFLOW_EXPERIMENT_NAME = 'MLOPS_PROJECT'
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
NUM_TUNE_TRIALS = 5
MODEL_NAME = 'resnet18'


caltech101_data = datasets.Caltech101(root='./temp', download=True)

train_dataset, test_dataset = train_val_split(caltech101_data)
_, val_dataset = train_val_split(train_dataset.dataset)


model_pipeline = ModelPipeline(MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI)
model = model_pipeline.get_model(MODEL_NAME)

if not model:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_final_layer = model.fc.in_features
    model.fc = nn.Linear(in_final_layer, len(caltech101_data.categories))


search_space = {
    'optim':{
        'lr': uniform_space(1e-4, 1e-2),
        'weight_decay': uniform_space(1e-6, 1e-3)
    },
    'batch_size': random_choice([32, 64]),
}

model_pipeline = ModelPipeline(MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI)
best_config = model_pipeline.tune(model, val_dataset, NUM_TUNE_TRIALS, search_space)
model_pipeline.train(model, train_dataset, test_dataset, best_config, MODEL_NAME)