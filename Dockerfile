FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl \
    build-essential cmake ninja-build \
    libopenblas-dev \
    build-essential cmake ninja-build \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*


#python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

#Llama from source
# ENV CMAKE_ARGS="-DGGML_CUDA=on"
# RUN pip install llama-cpp-python --no-binary llama-cpp-python --force-reinstall --no-cache-dir
# CUDA linker
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
# RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
# RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-binary llama-cpp-python --force-reinstall --no-cache-dir

#Llama from source - from colab
RUN pip install scikit-build-core==0.9.0
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=75"
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs
ENV FORCE_CMAKE=1
RUN pip install llama-cpp-python==0.2.62 --no-binary llama-cpp-python \
    --force-reinstall --upgrade --no-cache-dir --verbose --no-build-isolation


# Python deps

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
Casa
# Build and install llama-cpp-python from source with CUDA (cuBLAS) support
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Copy FastAPI app
COPY app.py /app/app.py
WORKDIR /app

# # Create model directory
# RUN mkdir -p /app/models
# COPY models /app/models

#Hugging Face
# RUN mkdir -p /app/models/stheno
# # Set build arg
# ARG HF_TOKEN
# ENV HF_TOKEN=${HF_TOKEN}
# RUN mkdir -p /app/models/stheno
# # Set build arg
# ARG HF_TOKEN
# ENV HF_TOKEN=${HF_TOKEN}

# # Authenticate and download the model
# RUN echo "🔐 Logging into Hugging Face..." && \
#     huggingface-cli login --token ${HF_TOKEN} && \
#     if [ ! -f /app/models/stheno/L3-8B-Stheno-v3.2-Q4_K_M.gguf ]; then \
#       huggingface-cli download bartowski/L3-8B-Stheno-v3.2-GGUF \
#         --include "L3-8B-Stheno-v3.2-Q4_K_M.gguf" \
#         --local-dir /app/models/stheno; \
#     else \
#       echo "✅ Model already exists, skipping download."; \
#     fi

WORKDIR /app

# Expose FastAPI port
EXPOSE 8000

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["/bin/bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000"]

