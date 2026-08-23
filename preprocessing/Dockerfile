#FROM apache/beam_python3.12_sdk:2.76.0rc2
FROM scratch
# Install system dependencies, CUDA repo keys, and gcsfuse
# Install FUSE and the official Google gcsfuse tool
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    fuse \
    ca-certificates \
    && echo "deb https://google.com gcsfuse-focal main" | tee /etc/apt/sources.list.d/gcsfuse.list \
    && curl https://google.com/doc/apt-key.gpg | apt-key add - \
    && apt-get update && apt-get install -y gcsfuse \
    && rm -rf /var/lib/apt/lists/*

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
