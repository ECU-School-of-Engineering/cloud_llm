FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set working directory
WORKDIR /app

# Expose API port
EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80", "--reload"]