import torch.nn as nn
from torchvision import datasets, models

from training.models.helpers.model_pipeline import ModelPipeline
from training.models.helpers.ray_helper import uniform_space, random_choice

from training.models.helpers.data_helper import train_dataset, val_dataset, test_dataset, label_to_idx   

MLFLOW_EXPERIMENT_NAME = 'MLOPS_PROJECT'
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
NUM_TUNE_TRIALS = 5



model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
in_final_layer = model.classifier.in_features
model.classifier = nn.Linear(in_final_layer, len(label_to_idx))  # Adjust the final layer for the number of classes

search_space = {
    'optim':{
        'lr': uniform_space(1e-4, 1e-2),
        'weight_decay': uniform_space(1e-6, 1e-3)
    },
    'batch_size': random_choice([32, 64]),
}

model_pipeline = ModelPipeline(MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI)

best_config = model_pipeline.tune(model, train_dataset, val_dataset, label_to_idx, NUM_TUNE_TRIALS, search_space)
model_pipeline.train(model, train_dataset, val_dataset, test_dataset, label_to_idx, best_config, 'mobilenet_v3_small')