from pathlib import Path
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Food Nutrition ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use system ephemeral tmp directory on Cloud Run
TEMP_DIR = Path("/tmp/temp_uploads")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def health_check():
    """Health check endpoint for Cloud Run/monitoring."""
    return {"status": "healthy", "service": "Food Nutrition ML API"}


@app.post("/analyze-food")
async def analyze_food(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    temp_path = TEMP_DIR / file.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Defer heavy ML pipeline import until request execution
        # This keeps container startup under 1 second for Cloud Run health checks
        from inference.pipeline import process_image

        result = process_image(temp_path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path.exists():
            os.remove(temp_path)