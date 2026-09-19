# Production Dockerfile for Fruit Fly Connectome & Bio-AI Suite
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt-get/lists/*

# Copy requirements & install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose port for FastAPI backend & Web portal
EXPOSE 8000
EXPOSE 8990

# Default entrypoint: Run FastAPI server
CMD ["uvicorn", "connectome_products.4_pharma_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
