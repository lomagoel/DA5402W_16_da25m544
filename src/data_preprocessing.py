

from asyncio.log import logger
from typing import Counter

from torch import cat
from .common import label_map

def data_cleaning(image_paths: list[str] | list[Path]) -> tuple[list[str], list[str]]:
    """Validate images by checking if they can be opened without corruption.
    Args:
      image_paths: List of image file paths to validate.
    Returns:
      Tuple containing two lists: (valid_images, corrupted_images)
    """
    # image can be opened and is not corrupted
    # output a list of valid images and a list of corrupted images
    # run the function in parallel threads
    valid_images = []
    corrupted_images = []
    labels_id = []
    labels_name=[]
    
        
    for img_path in image_paths:
        image = cv2.imread(img_path)
        label_name = Path(img_path).parent.name
        label_idx = label_map[label_name]
        labels_id.append(label_idx)
        labels_name.append(label_name)
        if image is None:
            corrupted_images.append(img_path)
        else:
            valid_images.append(img_path)

    logger.info(f"Found {len(corrupted_images)} corrupted images out of {len(image_paths)} total images.")
    logger.info(f"Found {len(valid_images)} valid images out of {len(image_paths)} total images.")
    logger.info(f"Found {len(set(labels_id))} unique labels out of {len(image_paths)} total images.")

    logger.info(f"---+Counter(labels_name)---+")
    logger.info(Counter(labels_name))
    return valid_images, corrupted_images, labels_id, labels_name


def class_imbalace_check(labels_id) -> None:
    # check if the ratio of highest class to lowest class is greater than 2
    # if so, log warning
    counter = Counter(labels_id)
    max_count = max(counter.values())
    min_count = min(counter.values())
    if max_count / min_count > 2:
        logger.warning(f"Class imbalance detected: {max_count} / {min_count} = {max_count / min_count:.2f}")
        logger.info(f"Class distribution: {dict(counter)}")


    return None

