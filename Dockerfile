# take pytorch with cuda image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# image lib update
apt-get update && apt-get install -y --no-install-recommends \
curl && \ 
rm -rf /var/lib/apt/lists/* # clear downloaded package metadata

# set workingdir inside which all code will be saved
WORKDIR /app

# install requirements
COPY ./reqirements.txt .
pip install --no-cache-dir -r requirements.txt

# copy content 
COPY --parents ./src/* ./scripts/* .


ENTRYPOINT ["python","train.py"]
