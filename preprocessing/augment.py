from glob import glob
import io
import os
import apache_beam as beam
from apache_beam.io import fileio  
from apache_beam import Pipeline
from PIL import Image
import torchvision.transforms as T
import hashlib
# Create a global cache of sorted files per class to enable deterministic index filtering
import glob
from collections import defaultdict

in_gcs_root = "gs://dataset_mtech/raw_caltech-101/"
out_gcs_root = "gs://dataset_mtech/caltech-101/"

all_files = glob.glob(os.path.join(in_gcs_root, "*", "*.jpg"))

# Group and sort files by class folder name
class_file_map = defaultdict(list)
for f in all_files:
    lbl = f.split('/')[-2]
    class_file_map[lbl].append(f)

for lbl in class_file_map:
    class_file_map[lbl].sort()  # Sort alphabetically to keep it deterministic

def get_stratified_split_filter(target_partition):
    """
    Guarantees 100% class distribution by forcing the first 3 items of 
    ANY class into train(0), val(1), and test(2) respectively.
    """
    def _filter(image_path):
        label = image_path.split('/')[-2]
        filename = os.path.basename(image_path)
        
        # Find the absolute deterministic index of this file within its own class folder
        try:
            file_index = class_file_map[label].index(image_path)
        except ValueError:
            file_index = 999  # Fallback safety
            
        # GUARANTEE RULE: Force representation for the first 3 items of every class
        if file_index == 0:
            current_partition = 0  # Force 1st item to Train
        elif file_index == 1:
            current_partition = 1  # Force 2nd item to Val
        elif file_index == 2:
            current_partition = 2  # Force 3rd item to Test
        else:
            # Fallback to standard 80/10/10 hash splitting for the rest of the images
            hash_digest = hashlib.md5(f"{label}_{filename}".encode('utf-8')).hexdigest()
            hash_int = int(hash_digest, 16)
            percentage = hash_int % 100
            
            if percentage < 80:
                current_partition = 0
            elif 80 <= percentage < 90:
                current_partition = 1
            else:
                current_partition = 2
                
        return current_partition == target_partition
    return _filter

class PreProcess(beam.DoFn):
    def __init__(self, is_train=True):
        self.is_train = is_train

    def setup(self):
        if self.is_train:
            self.augment = T.Compose([
                T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
            ])
        else:
            self.augment = T.Compose([
                T.Resize(size=(224, 224))
            ])

    def process(self, image_path):
        try:
            pil_img = Image.open(image_path).convert('RGB')
            augmented_img = self.augment(pil_img)

            byte_arr = io.BytesIO()
            augmented_img.save(byte_arr, format='JPEG', quality=95)
            jpeg_bytes = byte_arr.getvalue()
            
            yield (image_path, jpeg_bytes)
                
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

class DirectImageWriter(beam.DoFn):
    def __init__(self, base_output_path):
        self.base_output_path = base_output_path

    def process(self, element, window=beam.DoFn.WindowParam):
        image_path, jpeg_bytes = element
        
        label = image_path.split('/')[-2]
        filename = os.path.basename(image_path)
        
        target_dir = os.path.join(self.base_output_path, label)
        os.makedirs(target_dir, exist_ok=True)
        
        final_path = os.path.join(target_dir, filename)
        
        with open(final_path, 'wb') as f:
            f.write(jpeg_bytes)
            
        yield final_path

with beam.Pipeline() as pipeline:
    # 1. Base files loader
    raw_images = (
        pipeline 
        | "Find Images" >> fileio.MatchFiles(in_gcs_root + "caltech-101/*/*.jpg")
        | "Read Matches" >> fileio.ReadMatches()
        | "Extract Data" >> beam.Map(lambda f: f.metadata.path)
    )
    
    # 2. RUN SEPARATELY FOR TRAIN (Target Partition: 0)
    _ = (
        raw_images
        | "Filter Train Slice" >> beam.Filter(get_stratified_split_filter(target_partition=0))
        | "Reshuffle Train" >> beam.Reshuffle()
        | "Augment Train" >> beam.ParDo(PreProcess(is_train=True))
        | "Direct Disk Write Train" >> beam.ParDo(DirectImageWriter(base_output_path=os.path.join(out_gcs_root, "train")))
    )

    # 3. RUN SEPARATELY FOR VALIDATION (Target Partition: 1)
    _ = (
        raw_images
        | "Filter Val Slice" >> beam.Filter(get_stratified_split_filter(target_partition=1))
        | "Reshuffle Val" >> beam.Reshuffle()
        | "Augment Val" >> beam.ParDo(PreProcess(is_train=False))
        | "Direct Disk Write Val" >> beam.ParDo(DirectImageWriter(base_output_path=os.path.join(out_gcs_root, "val")))
    )

    # 4. RUN SEPARATELY FOR TEST (Target Partition: 2)
    _ = (
        raw_images
        | "Filter Test Slice" >> beam.Filter(get_stratified_split_filter(target_partition=2))
        | "Reshuffle Test" >> beam.Reshuffle()
        | "Augment Test" >> beam.ParDo(PreProcess(is_train=False))
        | "Direct Disk Write Test" >> beam.ParDo(DirectImageWriter(base_output_path=os.path.join(out_gcs_root, "test")))
    )

# Execution completes here before calculating folder summary counts
splits = ["train", "val", "test"]

print("\n=== DATASET SPLIT COUNTS ===")
for split in splits:
    split_dir = os.path.join(out_gcs_root, split)
    images = glob(split_dir+ "/*/*.jpg")
    categories = set(os.path.basename(os.path.dirname(p)) for p in images)
    print(f"Split: {split:<6} | Total Images: {len(images):<5} | Unique Classes: {len(categories)}")
