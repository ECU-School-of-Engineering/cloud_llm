FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Make 'python' point to 'python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python requirements (except llama-cpp-python)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Build and install llama-cpp-python from source with CUDA (cuBLAS) support
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Copy FastAPI app
COPY app.py /app/app.py
WORKDIR /app

# Create and copy model files
RUN mkdir -p /app/models
COPY models /app/models

# Hugging Face model folder
RUN mkdir -p /app/models/stheno

# Build arg for Hugging Face token
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Authenticate and download GGUF model from Hugging Face
RUN echo "🔐 Logging into Hugging Face..." && \
    huggingface-cli login --token ${HF_TOKEN} && \
    if [ ! -f /app/models/stheno/L3-8B-Stheno-v3.2-Q4_K_M.gguf ]; then \
      huggingface-cli download bartowski/L3-8B-Stheno-v3.2-GGUF \
        --include "L3-8B-Stheno-v3.2-Q4_K_M.gguf" \
        --local-dir /app/models/stheno; \
    else \
      echo "✅ Model already exists, skipping download."; \
    fi

WORKDIR /app

# Expose FastAPI port
EXPOSE 8000

# Start the server with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
