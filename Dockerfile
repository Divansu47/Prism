# Use official lightweight Python image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for OpenCV, EasyOCR, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirement files first for layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project codebase
COPY . .

# Expose port (Cloud Run sets PORT env variable dynamically, default 8080)
EXPOSE 8080

# Command to run the FastAPI app with Uvicorn
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1