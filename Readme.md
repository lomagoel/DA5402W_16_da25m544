coomandsd use d: sudo snap install dvc --classic
# 1. Clean the package list cache
sudo apt-get clean

# 2. Re-verify the Google Cloud GPG signature key
curl https://google.com | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

# 3. Securely write the correct repository path 
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://google.com cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

# 4. Update the package database and install
sudo apt-get update && sudo apt-get install -y google-cloud-cli


## create service account in google
gcloud iam service-accounts create my-service-account \
    --display-name="data_manager" \
    --project=proud-apogee-502506-s3

gcloud iam service-accounts keys create gcp-key.json --iam-account=my-service-account@proud-apogee-502506-s3.iam.gserviceaccount.com

## creating dataset
- This creates the label map and splits the Caltech 101 dataset

1. Download caltech101 dataset
2. Run `data_division.py`


# apply through codex
move tr

# augment
source env/preprocessing/bin/activate
python3 augment.py 