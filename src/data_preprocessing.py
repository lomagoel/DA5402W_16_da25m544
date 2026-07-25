from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from .common import can_read_image_lightweight, label_map


logger = logging.getLogger(__name__)


def data_cleaning(image_paths: list[str] | list[Path]) -> tuple[list[str], list[str], list[int], list[str]]:
    valid_images: list[str] = []
    corrupted_images: list[str] = []
    labels_id: list[int] = []
    labels_name: list[str] = []

    for image_path in image_paths:
        image_path_str = str(image_path)
        label_name = Path(image_path_str).parent.name

        if label_name not in label_map:
            logger.warning('Skipping image with unknown label folder: %s', image_path_str)
            continue

        labels_id.append(label_map[label_name])
        labels_name.append(label_name)

        if can_read_image_lightweight(image_path_str):
            valid_images.append(image_path_str)
        else:
            corrupted_images.append(image_path_str)

    total_images = len(image_paths)
    logger.info('Found %d corrupted images out of %d total images.', len(corrupted_images), total_images)
    logger.info('Found %d valid images out of %d total images.', len(valid_images), total_images)
    logger.info('Found %d unique labels out of %d total images.', len(set(labels_id)), total_images)
    logger.info('Class counts: %s', Counter(labels_name))

    return valid_images, corrupted_images, labels_id, labels_name


def class_imbalace_check(labels_id: list[int]) -> None:
    if not labels_id:
        logger.warning('No labels provided for imbalance check.')
        return

    counter = Counter(labels_id)
    max_count = max(counter.values())
    min_count = min(counter.values())

    if min_count == 0:
        logger.warning('At least one class has zero samples.')
        return

    ratio = max_count / min_count
    if ratio > 2:
        logger.warning('Class imbalance detected: %d / %d = %.2f', max_count, min_count, ratio)
        logger.info('Class distribution: %s', dict(counter))
