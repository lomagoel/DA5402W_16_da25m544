#!/bin/bash

# ==========================================
# Configuration Variables
# ==========================================
LOCAL_FOLDER_PATH="./training_dataset/data"      # The local folder to upload
GCS_BUCKET_URI="gs://caltech_1000/data"  # The destination bucket URI
SERVICE_ACCOUNT_JSON="./gcp-key.json"   # @TODO: shift to secrests

# ==========================================
# Execution
# ==========================================
echo "Starting upload process to GCS..."

# 1. Check if the local directory exists
if [ ! -d "$LOCAL_FOLDER_PATH" ]; then
    echo "Error: Local directory '$LOCAL_FOLDER_PATH' does not exist."
    exit 1
fi

# 2. Authenticate with Google Cloud silently (Required for Airflow/Automation)
echo "Authenticating via Service Account..."
gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_JSON"

if [ $? -ne 0 ]; then
    echo "Error: Failed to authenticate with GCP. Check your JSON key path."
    exit 1
fi

# clean previous training data 
gsutil rm -r gs://caltech_1000/data

# 3. Upload the folder using multithreading (-m) for faster transfers
# Using 'rsync' ensures it only uploads new or changed files, saving bandwidth
echo "DRY RUN"
gcloud storage rsync "$LOCAL_FOLDER_PATH" "$GCS_BUCKET_URI" --recursive  --dry-run


echo "Syncing folder to $GCS_BUCKET_URI..."
gcloud storage rsync "$LOCAL_FOLDER_PATH" "$GCS_BUCKET_URI" --recursive 

# 4. Check if the upload was successful
if [ $? -eq 0 ]; then
    echo "Success! Folder successfully uploaded to GCS."
else
    echo "Error: Failed to upload files to GCS."
    exit 1
fi