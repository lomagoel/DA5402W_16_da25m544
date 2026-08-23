FROM apache/beam_python3.12_sdk:2.76.0rc2

# Install system dependencies, CUDA repo keys, and gcsfuse
# Install FUSE and the official Google gcsfuse tool
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    fuse \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y curl lsb-release && \
    export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s) && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt ${GCSFUSE_REPO} main" \
        > /etc/apt/sources.list.d/gcsfuse.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        > /usr/share/keyrings/cloud.google.asc && \
    apt-get update && \
    apt-get install -y gcsfuse
    
# Set working directory inside which all code will be saved
WORKDIR /app

# Copy application content
COPY ./augment.py .   
COPY ./requirements.txt .


# Install your extra requirements
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

RUN ls -la /app

# Copy and prepare the startup entrypoint script
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
