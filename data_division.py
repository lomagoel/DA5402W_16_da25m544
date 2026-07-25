from __future__ import annotations

# load files from dolfer home/amol/Downloads/caltech-101/
# it has images of different categories. You can use the following code to load the images and their corresponding labels into a dataset
# prepare five parts: training set 1 (20%) .. training set 5 (20%) and a validation set (20%). You can use the following code to split the dataset into these parts:

def divide_dataset():
    # divide the dataset into num_parts equal parts part1/category1, part2 ..
    # return a list of datasets
    # read each folder home/amol/Downloads/caltech-101/category1
    return None

# load files from dolfer home/amol/Downloads/caltech-101/
# it has images of different categories. You can use the following code to load the images and their corresponding labels into a dataset
# prepare five parts: training set 1 (20%) .. training set 5 (20%) and a validation set (20%). You can use the following code to split the dataset into these parts:
"""Dataset preparation utilities for Caltech-101.

def divide_dataset():
    # divide the dataset into num_parts equal parts part1/category1, part2 ..
    # return a list of datasets
    # read each folder home/amol/Downloads/caltech-101/category1
Assumes the following on-disk layout:

  ~/Downloads/caltech-101/<category_name>/*.jpg|*.png|...

Each category is split independently (stratified) so every train/val split has
roughly the same per-class distribution.
"""


import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass(frozen=True)
class Split:
    """A split holding (image_path, label) pairs."""

    items: List[Tuple[str, int]]


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED_EXTS


def _collect_by_category(root_dir: Path) -> Dict[str, List[str]]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Caltech-101 root not found: {root_dir}")

    by_cat: Dict[str, List[str]] = {}
    for cat_dir in sorted(root_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        imgs = [str(p) for p in sorted(cat_dir.iterdir()) if _is_image_file(p)]
        if imgs:
            by_cat[cat_dir.name] = imgs
    if not by_cat:
        raise RuntimeError(f"No categories/images found under: {root_dir}")
    return by_cat


def divide_dataset(
    caltech_root: str | os.PathLike = os.path.expanduser("~/Downloads/archive/caltech-101"),
    *,
    num_train_parts: int = 5,

    seed: int = 42,
) -> Tuple[List[Split], Split, Dict[str, int]]:
    """Divide Caltech-101 into 5 training folds (20% each) and 20% validation.

    Returns:
      - train_splits: list of length `num_train_parts`, each split is 20% of
        every class (for the default parameters).
      - val_split: 20% of every class.
      - label_map: mapping from category_name -> integer label.

    Notes:
      - Splitting is stratified per category.
      - Deterministic given `seed`.
      - Requires `val_fraction == 1 / (num_train_parts + 1)` for equal 20% parts.
    """



    root_dir = Path(caltech_root)
    by_cat = _collect_by_category(root_dir)
    categories = sorted(by_cat.keys())
    label_map = {cat: i for i, cat in enumerate(categories)}

    rng = random.Random(seed)

    # name folder caltech-1, caltech-2, caltech-3, caltech-4, caltech-5
    # save 20% of each category in each folder
    train_splits: List[Split] = [Split([]) for _ in range(num_train_parts)]
    val_split = Split([])

    for cat, imgs in by_cat.items():
        rng.shuffle(imgs)
        num_imgs = len(imgs)
        num_train_per_part = (num_imgs ) // num_train_parts


        # Assign training images to each part
        start_idx = 0
        for part_idx in range(num_train_parts):
            end_idx = start_idx + num_train_per_part
            if part_idx == num_train_parts - 1:
                # Last part takes any remaining images due to rounding
                end_idx = num_imgs
            train_splits[part_idx].items.extend(
                (img, label_map[cat]) for img in imgs[start_idx:end_idx]
            )
            start_idx = end_idx

    # save labelmap as a text file in the output directory
    output_dir = Path(os.path.expanduser("~/Downloads/caltech_splits"))
    output_dir.mkdir(parents=True, exist_ok=True)
    labelmap_path = output_dir / "label_map.txt"
    with open(labelmap_path, "w") as f:
        for cat, idx in label_map.items():
            f.write(f"{cat} {idx}\n")

    return train_splits, val_split, label_map

def save_splits(
    train_splits: List[Split],
    val_split: Split,
    label_map: Dict[str, int],
    output_dir: str | os.PathLike = os.path.expanduser("~/Downloads/caltech_splits"),
) -> None:
    """Save the splits to disk as image files.
    Save image instead of text files.
    Each split is saved as a folder containing the images.
    Args:
      train_splits: List of training splits.
      val_split: Validation split.
      label_map: Mapping from category name to integer label.       """
    
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training splits
    for part_idx, split in enumerate(train_splits):
        part_dir = output_dir / f"train_part_{part_idx + 1}"
        part_dir.mkdir(exist_ok=True)
        for img_path, label in split.items:
            cat_name = next(cat for cat, idx in label_map.items() if idx == label)
            cat_dir = part_dir / cat_name
            cat_dir.mkdir(exist_ok=True)
            dest_path = cat_dir / Path(img_path).name
            os.link(img_path, dest_path)  # Create a hard link to save space

    # Save validation split
    val_dir = output_dir / "val"
    val_dir.mkdir(exist_ok=True)
    for img_path, label in val_split.items:
        cat_name = next(cat for cat, idx in label_map.items() if idx == label)
        cat_dir = val_dir / cat_name
        cat_dir.mkdir(exist_ok=True)
        dest_path = cat_dir / Path(img_path).name
        os.link(img_path, dest_path)  # Create a hard link to save space


if __name__ == "__main__":
    train_splits, val_split, label_map = divide_dataset()
    save_splits(train_splits, val_split, label_map)