import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchvision.transforms.v2 import Compose, Resize, RandomHorizontalFlip, ToImage, ToDtype, Lambda

import glob
import io
import torch
from torch.utils.data import IterableDataset, DataLoader

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


def dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



