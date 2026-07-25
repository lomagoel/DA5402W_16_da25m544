from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError


def load_label_map(label_map_path: str | Path = Path(__file__).resolve().parent.parent / 'label_map.txt') -> dict[str, int]:
    label_map: dict[str, int] = {}
    with open(label_map_path, 'r', encoding='utf-8') as file_handle:
        for line in file_handle:
            category_name, category_index = line.strip().split()
            label_map[category_name] = int(category_index)
    return label_map


def can_read_image_lightweight(image_path: str | Path) -> bool:
    try:
        with Image.open(image_path) as image_handle:
            image_handle.verify()
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
        return False


label_map = load_label_map()
