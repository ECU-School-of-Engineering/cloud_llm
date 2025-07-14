FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

#python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app
COPY app.py /app/app.py
WORKDIR /app

# Create model directory
RUN mkdir -p /app/models
COPY models /app/models

#Hugging Face
RUN mkdir -p /app/models/stheno
# Set build arg
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Authenticate and download the model
RUN echo "🔐 Logging into Hugging Face..." && \
    echo "${HF_TOKEN}" | huggingface-cli login --token && \
    huggingface-cli download bartowski/L3-8B-Stheno-v3.2-GGUF \
      --include "L3-8B-Stheno-v3.2-Q4_K_M.gguf" \
      --local-dir /app/models/stheno

WORKDIR /app

# Expose API port
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
