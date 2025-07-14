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

#Hugging Face

# Copy app
COPY app.py /app/app.py
WORKDIR /app

# Create model directory
RUN mkdir -p /app/models
COPY models /app/models

# Expose API port
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
