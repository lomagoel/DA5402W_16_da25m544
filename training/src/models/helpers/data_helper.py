import yaml
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchvision.transforms.v2 import Compose, Resize, RandomHorizontalFlip, ToImage, ToDtype
from torchvision import datasets

import glob
import torch
from torch.utils.data import IterableDataset, DataLoader


train_transforms = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(),
    ToImage(),
    ToDtype(torch.float32, scale=True)
])

test_transforms = Compose([
    Resize((224, 224)),
    ToImage(),
    ToDtype(torch.float32, scale=True)
])


class BeamShardDataset(IterableDataset):
    """Streams images sequentially out of multi-record Apache Beam shards."""
    def __init__(self, shard_pattern, label_to_idx):
        # 1. This finds the shard files (e.g., caltech_shard-00001-of-00020)
        self.shard_paths = glob.glob(shard_pattern)
        self.label_to_idx = label_to_idx

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Split shards among PyTorch DataLoader workers to avoid duplicate processing
        if worker_info is None:
            shards_to_process = self.shard_paths
        else:
            # Simple round-robin assignment per worker process
            shards_to_process = [
                s for i, s in enumerate(self.shard_paths) 
                if i % worker_info.num_workers == worker_info.worker_id
            ]

        # 2. Loop through each shard file assigned to this worker
        for shard_path in shards_to_process:
            with open(shard_path, 'rb') as f:
                # 3. Read individual image records out of the open shard
                while True:
                    try:
                        # Load the next serialized tensor/label dictionary packet
                        data = torch.load(f, weights_only=False)
                        
                        image_tensor = data['image']
                        label_name = data['label']
                        label_idx = self.label_to_idx[label_name]
                        
                        # 4. Yield one image-label pair to the DataLoader batch
                        yield image_tensor, label_idx
                        
                    except EOFError:
                        # Reached the end of this specific shard file, move to next
                        break
                    except Exception as e:
                        # Handle or skip corrupted items within a shard
                        continue

label_to_idx={}
with open("./label_map.txt", "r") as f:
    for line in f:
        label_name, idx = line.strip().split()
        label_to_idx[label_name] = int(idx)

gcs_root = "/gcs/dataset_mtech/"
# 1. Create the iterable dataset pointing to Beam outputs
train_dataset = BeamShardDataset(
    shard_pattern=gcs_root + "train/shard/*.pt",
    label_to_idx=label_to_idx
)
val_dataset = BeamShardDataset(
    shard_pattern=gcs_root + "val/shard/*.pt",
    label_to_idx=label_to_idx
)
test_dataset = BeamShardDataset(
    shard_pattern=gcs_root + "test/shard/*.pt",
    label_to_idx=label_to_idx
)


def dataloader(dataset, batch_size, shuffle, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def get_datasets(storage_mount_path):
    train_dataset = datasets.ImageFolder(root=f'{storage_mount_path}/train', transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=f'{storage_mount_path}/val', transform=test_transforms)
    test_dataset = datasets.ImageFolder(root=f'{storage_mount_path}/test', transform=test_transforms)
    return train_dataset, val_dataset, test_dataset


def train_val_split(data, test_size=0.2, apply_stratify=True):
    target = data.classes
    if apply_stratify:
        train_idx, test_idx = train_test_split(np.arange(len(target)), test_size=test_size, shuffle=True, stratify=target, random_state=42)
    else:
        train_idx, test_idx = train_test_split(np.arange(len(target)), test_size=test_size, shuffle=True, random_state=42)

    train_dataset = Subset(data, train_idx)
    test_dataset = Subset(data, test_idx)
    return train_dataset, test_dataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)




'''
# Update local packages and install system requirements
sudo apt-get update
sudo apt-get install -y curl lsb-release

# Add the official Cloud Storage FUSE distribution URL as a package source
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list

# Import the Google Cloud Apt repository key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc

# Update package lists and install gcsfuse
sudo apt-get update
sudo apt-get install -y gcsfuse

# 1. Create a local directory (e.g., in your home directory or /mnt)
mkdir -p $HOME/my-bucket-mount

# 2. Mount your GCS bucket to that folder
# Replace 'your-bucket-name' with your actual GCP bucket name
gcsfuse --implicit-dirs your-bucket-name $HOME/my-bucket-mount

'''