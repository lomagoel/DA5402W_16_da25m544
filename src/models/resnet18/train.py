import torch.nn as nn
from torchvision import datasets, models

from src.models.helpers.model_pipeline import ModelPipeline
from src.models.helpers.ray_helper import uniform_space, random_choice

from src.models.helpers.data_helper import train_dataset, val_dataset, test_dataset, label_to_idx   


MLFLOW_EXPERIMENT_NAME = 'MLOPS_PROJECT'
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
NUM_TUNE_TRIALS = 5




model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_final_layer = model.fc.in_features
model.fc = nn.Linear(in_final_layer, len(label_to_idx))  # Adjust the final layer for the number of classes

search_space = {
    'optim':{
        'lr': uniform_space(1e-4, 1e-2),
        'weight_decay': uniform_space(1e-6, 1e-3)
    },
    'batch_size': random_choice([32, 64]),
}

model_pipeline = ModelPipeline(MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI)
best_config = model_pipeline.tune(model, val_dataset, NUM_TUNE_TRIALS, search_space)
model_pipeline.train(model, train_dataset, test_dataset, label_to_idx, best_config, 'resnet18')