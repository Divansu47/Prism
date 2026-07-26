import io
import logging
import os
from typing import Any, Dict, List

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from src.train_pipeline import DEFAULT_TARGET_COLUMNS, NutritionRegressor
from src.segmentation_pipeline.infer import SegmentationNutritionPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="Food Nutrition Inference API",
    description="Predict food composition and nutrition from images using segmentation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("weights", "nutrition_model.pth"))
SEGMENTATION_MODEL_PATH = os.environ.get("SEGMENTATION_MODEL_PATH", os.path.join("weights", "segmentation_model.pth"))
MODEL: NutritionRegressor | None = None
SEGMENTATION_MODEL: SegmentationNutritionPipeline | None = None


def _load_model() -> NutritionRegressor:
    global MODEL
    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        model = NutritionRegressor(num_outputs=len(DEFAULT_TARGET_COLUMNS))
        model.load_state_dict(checkpoint)
        model.eval()
        MODEL = model
    return MODEL


def _preprocess_image(image_bytes: bytes) -> torch.Tensor:
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Image file is empty.")
    if len(image_bytes) > 10 * 1024 * 1024:
        raise ValueError("Image file exceeds 10MB limit.")
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image format: {str(e)}")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def _load_segmentation_model() -> SegmentationNutritionPipeline:
    global SEGMENTATION_MODEL
    if SEGMENTATION_MODEL is None:
        if not os.path.exists(SEGMENTATION_MODEL_PATH):
            raise FileNotFoundError(f"Segmentation weights not found at {SEGMENTATION_MODEL_PATH}")
        SEGMENTATION_MODEL = SegmentationNutritionPipeline(
            model_path=SEGMENTATION_MODEL_PATH,
            class_names=["rice_staple", "curry_gravy", "vegetable", "protein"],
        )
    return SEGMENTATION_MODEL


@app.get("/health", tags=["System"])
def health() -> Dict[str, Any]:
    """Check API health and model availability."""
    status = {"status": "ok", "version": "1.0.0"}
    try:
        _load_segmentation_model()
        status["segmentation_model"] = "ready"
    except Exception as e:
        logger.warning(f"Segmentation model not ready: {e}")
        status["segmentation_model"] = "unavailable"
    return status


@app.post("/predict", tags=["Inference"])
def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Predict nutrition from a food bowl image (direct regression).
    
    Returns 34 nutrition columns including calories, macros, vitamins, minerals.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        image_bytes = file.file.read()
        _preprocess_image(image_bytes)
        tensor = _preprocess_image(image_bytes)
        model = _load_model()
        with torch.no_grad():
            preds = model(tensor).squeeze(0).cpu().tolist()
        logger.info(f"Prediction successful for {file.filename}")
    except FileNotFoundError as exc:
        logger.error(f"Model not found: {exc}")
        raise HTTPException(status_code=503, detail="Model unavailable. Please try again later.") from exc
    except ValueError as exc:
        logger.warning(f"Invalid image: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}") from exc

    return {
        "status": "success",
        "columns": DEFAULT_TARGET_COLUMNS,
        "predictions": preds,
    }


@app.post("/predict_segmentation", tags=["Inference"])
def predict_segmentation(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Predict food composition and nutrition using segmentation (recommended).
    
    Detects food regions (rice, curry, vegetable, protein) and returns:
    - Detected regions with confidence scores
    - Estimated nutrition profile per 100g of detected composition
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    image_path = None
    try:
        image_bytes = file.file.read()
        _preprocess_image(image_bytes)
        image_path = os.path.join("weights", "uploaded_image_temp.png")
        with open(image_path, "wb") as fh:
            fh.write(image_bytes)
        model = _load_segmentation_model()
        result = model.predict(image_path)
        logger.info(f"Segmentation prediction successful for {file.filename}")
        result["status"] = "success"
        return result
    except FileNotFoundError as exc:
        logger.error(f"Model not found: {exc}")
        raise HTTPException(status_code=503, detail="Model unavailable. Please try again later.") from exc
    except ValueError as exc:
        logger.warning(f"Invalid image: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Segmentation prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}") from exc
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)


@app.get("/docs_custom", tags=["System"])
def get_docs() -> Dict[str, Any]:
    """API documentation and usage guide."""
    return {
        "endpoints": {
            "/health": "Check API health and model readiness.",
            "/predict": "Direct nutrition regression from image (34 columns).",
            "/predict_segmentation": "Segmentation-based prediction with food regions and composition.",
        },
        "recommended_endpoint": "/predict_segmentation (more interpretable, shows food breakdown)",
        "models": {
            "segmentation_model_path": "weights/segmentation_model.pth",
            "nutrition_model_path": "weights/nutrition_model.pth",
        },
        "environment_variables": {
            "SEGMENTATION_MODEL_PATH": "Override default segmentation model path.",
            "MODEL_PATH": "Override default nutrition model path.",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
