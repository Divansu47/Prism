FROM python:3.11-slim

WORKDIR /app

# Install system libraries required by OpenCV
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Download model from Google Drive
RUN python download_models.py

# Environment variables
ENV PORT=8080
ENV SEGMENTATION_MODEL_PATH=weights/segmentation_model.pth

EXPOSE 8080

CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8080"]