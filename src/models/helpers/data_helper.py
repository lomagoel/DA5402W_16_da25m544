import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchvision.transforms.v2 import Compose, Resize, RandomHorizontalFlip, ToImage, ToDtype, Lambda


TRAIN_TRANSFORM = Compose([
    Resize((224, 224)),
    Lambda(lambda img: img.convert('RGB')),
    RandomHorizontalFlip(),
    ToImage(),
    ToDtype(torch.float32, scale=True)
])

TEST_TRANSFORM = Compose([
    Resize((224, 224)),
    Lambda(lambda img: img.convert('RGB')),
    ToImage(),
    ToDtype(torch.float32, scale=True)
])


def train_val_split(data, test_size=0.2):
    target = data.y
    train_idx, test_idx = train_test_split(np.arange(len(target)), test_size=test_size, shuffle=True, stratify=target, random_state=42)
    train_dataset = Subset(data, train_idx)
    test_dataset = Subset(data, test_idx)

    train_dataset.dataset.transform = TRAIN_TRANSFORM
    test_dataset.dataset.transform = TEST_TRANSFORM

    return train_dataset, test_dataset


def dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

