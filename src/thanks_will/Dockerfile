FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install verifiers library
RUN pip install 'verifiers[all]'

# Install other ML dependencies
RUN pip install \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    datasets==2.16.0 \
    tokenizers==0.15.0 \
    peft==0.7.1 \
    bitsandbytes==0.41.3 \
    openai \
    requests \
    numpy \
    pandas \
    wandb \
    tensorboard

# Set working directory
WORKDIR /workspace

# Copy your code
COPY . /workspace/

# Make sure the code is executable
RUN chmod +x /workspace/src/thanks_will/main.py

# Default command
CMD ["python", "/workspace/src/thanks_will/main.py"]