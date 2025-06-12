# Base image with Python 3.12 and CUDA support
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set working directory inside the container
WORKDIR /app

# Disable interactive prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    texlive-full \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file (if exists) and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/root/.local/bin:$PATH"

# Default command (interactive shell)
CMD ["python3"]
