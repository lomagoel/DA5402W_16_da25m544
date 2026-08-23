# take pytorch with cuda image
FROM docker pull apache/beam_python3.12_sdk:2.76.0rc2

# image lib update
RUN apt-get update && apt-get install -y --no-install-recommends \
curl && \ 
rm -rf /var/lib/apt/lists/* # clear downloaded package metadata

# set workingdir inside which all code will be saved
WORKDIR /app



# copy content 
COPY  ./augment.py ./src   

# install requirements
COPY .requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages


RUN ls -la /app


ENTRYPOINT ["python","augment.py"]

