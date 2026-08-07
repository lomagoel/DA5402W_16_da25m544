"""Task wrapper: download Caltech-101 from Kaggle, split, and upload to GCS.

Reads from environment:
  GCS_DATA_PATH     - GCS destination prefix  (e.g. gs://caltech_1000/data)
  KAGGLE_USERNAME   - Kaggle account username
  KAGGLE_KEY        - Kaggle API key
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from google.cloud import storage

from src.data_collection import divide_dataset, download_data, get_label_map, save_splits


def _upload_dir_to_gcs(local_dir: Path, gcs_prefix: str) -> None:
    """Recursively upload a local directory to a GCS prefix."""
    # gcs_prefix example: gs://caltech_1000/data
    if gcs_prefix.startswith("gs://"):
        gcs_prefix = gcs_prefix[5:]
    bucket_name, _, blob_prefix = gcs_prefix.partition("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for file_path in local_dir.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(local_dir)
        blob_name = f"{blob_prefix}/{relative}" if blob_prefix else str(relative)
        bucket.blob(blob_name).upload_from_filename(str(file_path))

    print(f"Uploaded {local_dir} → gs://{bucket_name}/{blob_prefix}")


def main() -> None:
    gcs_data_path = os.environ["GCS_DATA_PATH"]
    kaggle_username = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")

    # Patch Kaggle auth into data_collection if credentials provided via env
    if kaggle_username and kaggle_key:
        import src.data_collection as dc
        import requests
        from requests.auth import HTTPBasicAuth

        _orig_download = dc.download_data

        def _authed_download(data_dir):
            import zipfile
            url = "https://www.kaggle.com/api/v1/datasets/download/imbikramsaha/caltech-101"
            tmp = Path("./temp")
            tmp.mkdir(exist_ok=True)
            zip_path = tmp / "caltech.zip"
            if not zip_path.exists():
                print(f"Downloading Caltech-101 from {url} ...")
                resp = requests.get(
                    url,
                    stream=True,
                    verify=False,
                    auth=HTTPBasicAuth(kaggle_username, kaggle_key),
                )
                resp.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Extracting {zip_path} ...")
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(path=tmp)
            return tmp / "caltech-101"

        dc.download_data = _authed_download

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "split"

        caltech_root = download_data(output_dir)
        label_map = get_label_map(caltech_root)
        train_splits, val_split = divide_dataset(caltech_root)
        save_splits(train_splits, val_split, label_map, output_dir)

        # Write label_map.txt alongside the splits so it lands in GCS
        label_map_path = output_dir / "label_map.txt"
        with open(label_map_path, "w") as f:
            for cat, idx in label_map.items():
                f.write(f"{cat} {idx}\n")

        _upload_dir_to_gcs(output_dir, gcs_data_path)

    print(f"Data available at {gcs_data_path}")


if __name__ == "__main__":
    main()
