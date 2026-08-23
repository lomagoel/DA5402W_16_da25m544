import torch.nn as nn
from torchvision import models

from src.models.helpers.model_pipeline import ModelPipeline
from src.models.helpers.ray_helper import uniform_space, random_choice

from src.models.helpers.data_helper import get_datasets, load_config


config = load_config('src/training_config.yaml')
num_classes = config['data']['num_classes']
storage_mount_path = config['data']['storage_mount_path']
epochs = config['training']['epochs']

train_dataset, val_dataset, test_dataset = get_datasets(storage_mount_path)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_final_layer = model.fc.in_features
model.fc = nn.Linear(in_final_layer, num_classes)  # Adjust the final layer for the number of classes

search_space = {
    'optim':{
        'lr': uniform_space(1e-4, 1e-2),
        'weight_decay': uniform_space(1e-6, 1e-3)
    },
    'batch_size': random_choice([32, 64]),
}

model_pipeline = ModelPipeline(config['mlflow']['experiment_name'], config['mlflow']['tracking_uri'], num_classes)
best_config = model_pipeline.tune(model, val_dataset, config['tuning']['tune_trials'], search_space)
model_pipeline.train(model, train_dataset, val_dataset, test_dataset, best_config, epochs, 'resnet18')