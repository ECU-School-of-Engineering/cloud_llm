FROM python:3.10-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    git cmake build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app
COPY app.py /app/app.py
WORKDIR /app

# Create model directory
RUN mkdir -p /app/models

# Expose API port
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
