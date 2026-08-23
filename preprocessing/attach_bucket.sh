#!/bin/bash
# Stop script on any error
set -e

# 1. Define variables (Change 'your-bucket-name' to your actual GCS bucket)
BUCKET_NAME="dataset_mtech"
MOUNT_DIR="$HOME/gcs-bucket"

# 2. Add the official Cloud Storage FUSE distribution URL as a package source
export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list

# 3. Import the Google Cloud public signing key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc > /dev/null

# 4. Update repository lists and install gcsfuse
sudo apt-get update
sudo apt-get install -y gcsfuse

# 5. Configure FUSE to allow non-root users (highly recommended)
sudo sed -i 's/#user_allow_other/user_allow_other/' /etc/fuse.conf

# 6. Create the mount point directory inside your home directory
mkdir -p "$MOUNT_DIR"

# 7. Mount the bucket using standard user privileges
gcsfuse -o allow_other --implicit-dirs "$BUCKET_NAME" "$MOUNT_DIR"

echo "Success! Your bucket is mounted at $MOUNT_DIR"
