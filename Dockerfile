# take pytorch with cuda image
FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime

# image lib update
RUN apt-get update && apt-get install -y --no-install-recommends \
curl && \ 
rm -rf /var/lib/apt/lists/* # clear downloaded package metadata

# set workingdir inside which all code will be saved
WORKDIR /app

# install requirements
COPY ./reqirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy content 
COPY --parents ./src/* ./scripts/* .


ENTRYPOINT ["python","train.py"]
