FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
ENV SEGMENTATION_MODEL_PATH=weights/segmentation_model.pth
ENV MODEL_PATH=weights/nutrition_model.pth

EXPOSE 8080

CMD ["sh", "-c", "python download_models.py && uvicorn api.main:app --host 0.0.0.0 --port $PORT"]