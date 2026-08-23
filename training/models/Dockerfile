# take pytorch with cuda image
FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime

# image lib update
RUN apt-get update && apt-get install -y --no-install-recommends \
curl && \ 
rm -rf /var/lib/apt/lists/* # clear downloaded package metadata

# set workingdir inside which all code will be saved
WORKDIR /app



# copy content 
COPY  . .   

# install requirements
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages


RUN ls -la /app


ENTRYPOINT ["python","-m","models.resnet18"]

