"""Task wrapper: validate images and check class balance, write manifest to GCS.

Reads from environment:
  GCS_DATA_PATH      - GCS prefix where split data lives (e.g. gs://caltech_1000/data)
  GCS_MANIFEST_PATH  - GCS destination for the output manifest JSON
                       (e.g. gs://caltech_1000/manifests/clean_manifest.json)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from google.cloud import storage

from src.data_preprocessing import class_imbalace_check, data_cleaning


def _list_images_from_gcs(gcs_prefix: str) -> list[str]:
    """Return all image file paths under a GCS prefix as gs:// URIs."""
    if gcs_prefix.startswith("gs://"):
        gcs_prefix = gcs_prefix[5:]
    bucket_name, _, prefix = gcs_prefix.partition("/")

    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    return [
        f"gs://{bucket_name}/{blob.name}"
        for blob in blobs
        if Path(blob.name).suffix.lower() in image_exts
    ]


def _write_json_to_gcs(data: dict, gcs_path: str) -> None:
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]
    bucket_name, _, blob_name = gcs_path.partition("/")
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(
        json.dumps(data, indent=2), content_type="application/json"
    )


def main() -> None:
    gcs_data_path = os.environ["GCS_DATA_PATH"]
    gcs_manifest_path = os.environ["GCS_MANIFEST_PATH"]

    print(f"Listing images under {gcs_data_path} ...")
    image_paths = _list_images_from_gcs(gcs_data_path)
    print(f"Found {len(image_paths)} images.")

    valid_images, corrupted_images, labels_id, labels_name = data_cleaning(image_paths)
    class_imbalace_check(labels_id)

    manifest = {
        "gcs_data_path": gcs_data_path,
        "total": len(image_paths),
        "valid_count": len(valid_images),
        "corrupted_count": len(corrupted_images),
        "valid_images": valid_images,
        "corrupted_images": corrupted_images,
    }

    print(f"Writing manifest to {gcs_manifest_path} ...")
    _write_json_to_gcs(manifest, gcs_manifest_path)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
