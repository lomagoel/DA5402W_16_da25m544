# train mobilenet model on caltech101 dataset using mlflow
# train mobilenet model on caltech101 dataset using mlflow
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import mlflow
import mlflow.pytorch
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from src.common import label_map
import yaml 
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str 
    experiment_name: str 
    tracking_uri: str 
    epochs: int 
    batch_size: int 
    learning_rate: float 
    weight_decay: float 
    image_size: int 
    num_workers: int 
    seed: int 
    pretrained: bool 
    freeze_backbone: bool
    device: str
    class_to_idx: dict[str, int] = field(default_factory=lambda: label_map)
    

class CaltechFolderDataset(Dataset):
    """Image dataset for Caltech-style class-folder directories."""

    def __init__(
        self,
        roots: Iterable[Path],
        class_to_idx: dict[str, int],
        transform: transforms.Compose | None = None,
        corrupted_images: set[Path] | None = None,
    ) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.corrupted_images = corrupted_images or set()

        for root in roots:
            if not root.exists():
                raise FileNotFoundError(f"Dataset directory not found: {root}")
            for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                if class_dir.name not in class_to_idx:
                    continue
                label = class_to_idx[class_dir.name]
                for image_path in sorted(class_dir.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS and image_path not in self.corrupted_images:
                        self.samples.append((image_path, label))

        if not self.samples:
            root_list = ", ".join(str(root) for root in roots)
            raise RuntimeError(f"No images found in: {root_list}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label





def main() -> None:
    """Fine-tune the model. use mlflow to log the training process and save the model."""
    # read training config yaml
    with open("src/training_config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config = TrainConfig(**config_dict)
    train_loader = DataLoader(
        dataset=CaltechFolderDataset(
            roots=[Path(config.data_dir) / "train"],
            class_to_idx=config.class_to_idx,
            transform=transforms.Compose(
                [
                    transforms.Resize((config.image_size, config.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
        ),
    ),
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
)

    
    # load mobilenetv2 model with pretrained weights
    model = mobilenet_v2(pretrained=config.pretrained)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(config.class_to_idx))

    # move model to device
    model.to(config.device)

    # set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # train for the specified number of epochs
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{config.epochs}], Loss: {epoch_loss:.4f}")

        # step the scheduler
        scheduler.step()

    # evaluate 

if __name__ == "__main__":
    main()