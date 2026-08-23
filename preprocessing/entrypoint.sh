#!/bin/bash
set -e

if [ -z "$GCS_BUCKET" ]; then
    echo "ERROR: GCS_BUCKET environment variable is not set."
    exit 1
fi

echo "Mounting GCS bucket: $GCS_BUCKET..."
mkdir -p /mnt/gcs
gcsfuse --foreground "$GCS_BUCKET" /mnt/gcs &

# Wait for mount initialization
sleep 2

# Expose local path mappings to augment.py
export IN_GCS_ROOT="/mnt/gcs/raw_caltech-101/"
export OUT_GCS_ROOT="/mnt/gcs/caltech-101/"

echo "Executing Beam Data Augmentation Pipeline..."
python augment.py

# Unmount on finish
echo "Unmounting GCS bucket..."
fusermount -u /mnt/gcs
