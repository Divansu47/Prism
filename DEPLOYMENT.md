# Food Nutrition Inference API - Production Deployment Guide

## Overview

This is a production-ready FastAPI application that predicts food composition and nutrition from bowl images using deep learning segmentation.

## Features

- **CORS enabled** for cross-origin requests from web platforms
- **Structured error handling** with proper HTTP status codes (400, 503, 500)
- **Request validation** for image size (max 10MB) and format
- **Logging** for monitoring and debugging
- **Health checks** with model readiness status
- **Two prediction endpoints**:
  - `/predict`: Direct regression (34 nutrition columns)
  - `/predict_segmentation`: Segmentation-based with food region breakdown (recommended)

## API Endpoints

### GET `/health`
Check API and model availability.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "segmentation_model": "ready"
}
```

**Status codes:** 200 OK

---

### POST `/predict`
Direct nutrition prediction from image.

**Request:**
- Content-Type: multipart/form-data
- File: Image (JPG, PNG, WebP)

**Response (200 OK):**
```json
{
  "status": "success",
  "columns": [
    "Caloric Value",
    "Fat( in g)",
    ...
  ],
  "predictions": [270.0, 6.1, ...]
}
```

**Error responses:**
- 400: Bad image format or missing file
- 500: Internal error
- 503: Model not available

---

### POST `/predict_segmentation` (Recommended)
Segmentation-based prediction with food regions.

**Request:**
- Content-Type: multipart/form-data
- File: Image (JPG, PNG, WebP)

**Response (200 OK):**
```json
{
  "status": "success",
  "items": [
    {
      "class": "rice_staple",
      "category": "rice_staple",
      "confidence": 0.489
    },
    {
      "class": "vegetable",
      "category": "vegetable",
      "confidence": 0.517
    }
  ],
  "estimated_nutrition": {
    "Caloric Value": 270.0,
    "Protein( in g)": 10.8,
    "Carbohydrates( in g)": 43.8,
    ...
  }
}
```

**Error responses:** Same as `/predict`

---

## Deployment on Render

### 1. Prepare Environment Variables

Create a `.env` file or set in Render dashboard:

```env
SEGMENTATION_MODEL_PATH=weights/segmentation_model.pth
MODEL_PATH=weights/nutrition_model.pth
```

### 2. Create `render.yaml`

```yaml
services:
  - type: web
    name: food-nutrition-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port 8000
    envVars:
      - key: SEGMENTATION_MODEL_PATH
        value: weights/segmentation_model.pth
      - key: MODEL_PATH
        value: weights/nutrition_model.pth
```

### 3. Create `requirements.txt`

```
torch==2.0.1
torchvision==0.15.2
segmentation-models-pytorch==0.3.3
opencv-python==4.8.0.74
albumentations==1.3.0
pandas==2.0.3
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
Pillow==10.0.0
```

### 4. Push to GitHub and Connect to Render

1. Initialize git repo: `git init`
2. Add files: `git add .`
3. Commit: `git commit -m "Initial food nutrition API"`
4. Push to GitHub
5. Connect repo to Render at https://render.com
6. Deploy as Web Service

### 5. Test the Live Endpoint

```bash
curl -X POST https://your-service.onrender.com/health
curl -X POST https://your-service.onrender.com/predict_segmentation \
  -F "file=@my_bowl.jpg"
```

---

## Local Development

### Start the API

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access Interactive Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Test with cURL

```bash
# Health check
curl http://localhost:8000/health

# Predict with segmentation
curl -X POST http://localhost:8000/predict_segmentation \
  -F "file=@inference_test_bowl.jpg"
```

---

## Web Platform Integration

### JavaScript/TypeScript Example

```javascript
async function predictNutrition(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch('https://your-api.onrender.com/predict_segmentation', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return await response.json();
}

// Usage
const fileInput = document.getElementById('imageUpload');
fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  const result = await predictNutrition(file);
  console.log('Food components:', result.items);
  console.log('Estimated nutrition:', result.estimated_nutrition);
});
```

### Python Example

```python
import requests

def predict_nutrition(image_path: str, api_url: str):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f'{api_url}/predict_segmentation',
            files=files,
            timeout=30
        )
    return response.json()

# Usage
result = predict_nutrition('my_bowl.jpg', 'https://your-api.onrender.com')
print(f"Found: {result['items']}")
print(f"Nutrition: {result['estimated_nutrition']}")
```

---

## Model Files

Required files before deployment:

- `weights/segmentation_model.pth` - Trained segmentation model
- `weights/nutrition_model.pth` - Trained regression model (optional)
- `data/nutrition_db/food_lookup.json` - Kaggle nutrition database
- `data/nutrition_db/category_profile.json` - Per-category nutrition averages

If these files are missing, the API will return 503 (Service Unavailable) for prediction endpoints.

---

## Performance Notes

- Image processing: ~500ms
- Model inference: ~1-2s
- Total request time: 2-3s

For production, consider:
- GPU acceleration (if available)
- Request caching
- Batch processing
- Load balancing

---

## Support

For issues, check:
1. `/health` endpoint for model status
2. Server logs for detailed error messages
3. Image format (must be valid JPEG/PNG/WebP)
4. Image size (max 10MB)
