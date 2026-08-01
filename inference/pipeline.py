from pathlib import Path
import sys
import os
import cv2
import PIL.Image
from dotenv import load_dotenv
from google import genai
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load environment variables
load_dotenv()

# Initialize Gemini Client if API key exists
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# from segment import FoodSegmenter
# from depth import get_depth
# from volume import estimate_volume
# from nutrition import NutritionEstimator
# from mass import estimate_mass, scale_nutrition
# from dish_classifier import classify_dish
# from safety_checks import check_dominant_detection, check_dish_mismatch
# from ocr import detect_package_and_text

from .usda.search import search as usda_search
# Use explicit relative imports for modules inside the inference package
from .aliases import ALIASES
from .depth import get_depth
from .dish_classifier import classify_dish
from .geometry import *  # (or specific functions)
from .mass import estimate_mass, scale_nutrition
from .matcher import *
from .meal import *
from .nutrition import NutritionEstimator
from .ocr import detect_package_and_text
from .safety_checks import check_dish_mismatch, check_dominant_detection
from .segment import FoodSegmenter
from .utils import *
from .volume import estimate_volume
from .meal import (
    merge_foods,
    infer_meal,
    total_nutrition,
    total_geometry,
)

# MODEL_PATH = (
#     ROOT
#     / "runs"
#     / "segment"
#     / "runs"
#     / "segment"
#     / "runs"
#     / "foodseg103_rebalanced"
#     / "weights"
#     / "best.pt"
# )
MODEL_PATH = ROOT / "models" / "foodseg_best.pt"

# Global lazy-loaded segmenter to avoid re-loading weights per request in FastAPI
_SEGMENTER_INSTANCE = None

def get_segmenter():
    global _SEGMENTER_INSTANCE
    if _SEGMENTER_INSTANCE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model weights not found at: {MODEL_PATH}")
        _SEGMENTER_INSTANCE = FoodSegmenter(model_path=MODEL_PATH)
    return _SEGMENTER_INSTANCE


def call_gemini_vision_api(image_path, current_guess):
    if client is None:
        print("\nGemini API client not initialized. Skipping fallback.")
        return None
    try:
        img = PIL.Image.open(image_path)
        
        # Specialized prompt for complex multi-compartment tray breakdowns
        if current_guess == "complex_tray_breakdown":
            prompt = (
                "Analyze this food tray or meal image. "
                "List the distinct visible food items in this meal tray separated strictly by commas "
                "(e.g., Rice, Pinto Beans, Salsa, Roasted Potatoes, Salad). "
                "Do NOT include introductory text, bullet points, or extra explanation. "
                "Return ONLY the comma-separated names of the foods."
            )
        else:
            # Targeted prompt for single ingredient validation / correction
            prompt = (
                f"Analyze this food image. My local AI detected a specific ingredient: '{current_guess}'. "
                f"Please look closely for '{current_guess}' in the image. "
                f"If it is present (or something very similar is present like a garnish), simply return '{current_guess}'. "
                f"If the AI completely misclassified this specific item, return the true correct name of that specific item. "
                "Do NOT describe the entire plate. Focus ONLY on validating or correcting this one guessed ingredient. "
                "Return ONLY the food name, with no punctuation."
            )
            
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=[prompt, img]
        )
        return response.text.strip().lower()
    except Exception as e:
        print(f"\nGemini API Error: {e}")
        return None
def handle_packaged_product(image_path, ocr_result):
    query_text = ocr_result.get("query_text", "")
    if not query_text:
        return {
            "is_packaged": True,
            "status": "failed",
            "message": "No usable text extracted. Cannot search USDA database."
        }

    matches = usda_search(
        query_text,
        limit=5,
        prefer_generic=False,
        data_type="branded_food",
    )

    if not matches:
        matches = usda_search(
            query_text,
            limit=5,
            prefer_generic=False,
        )

    if not matches:
        return {
            "is_packaged": True,
            "status": "no_match",
            "query": query_text,
            "message": "No confident USDA match found for detected product text."
        }

    record = matches[0]
    overlap_ratio = record.get("match_overlap_ratio")
    fuzzy_ratio = record.get("match_fuzzy_ratio")
    warning = None

    if overlap_ratio is not None and overlap_ratio < 0.6:
        return {
            "is_packaged": True,
            "status": "low_confidence",
            "query": query_text,
            "matched_display_name": record.get("display_name"),
            "message": f"Low confidence match (token overlap: {overlap_ratio * 100:.0f}%). Verify manually."
        }

    if fuzzy_ratio is not None:
        warning = f"Matched via fuzzy correction (similarity: {fuzzy_ratio * 100:.0f}%)"

    nutrition_per_100g = {
        "calories": float(record["calories"]),
        "protein": float(record["protein"]),
        "fat": float(record["fat"]),
        "carbs": float(record["carbs"]),
        "fiber": float(record["fiber"]),
    }

    weight_grams = ocr_result.get("weight_grams")
    if weight_grams is not None:
        scaled_nutrition = scale_nutrition(nutrition_per_100g, weight_grams)
        # Ensure standard python floats
        nutrition = {k: float(v) for k, v in scaled_nutrition.items()}
    else:
        nutrition = nutrition_per_100g

    return {
        "is_packaged": True,
        "status": "success",
        "image": image_path.name,
        "query": query_text,
        "product_name": record.get("display_name"),
        "brand": record.get("brand_owner", "Unknown"),
        "data_type": record.get("data_type"),
        "warning": warning,
        "weight_detected_g": float(weight_grams) if weight_grams else None,
        "nutrition": nutrition
    }


def process_image(image_path: Path) -> dict:
    try:
        pil_img = PIL.Image.open(image_path).convert("RGB")
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return {"status": "error", "message": f"Could not read image {image_path}: {e}"}
        
    ocr_result = detect_package_and_text(image)
    if ocr_result["is_packaged"]:
        return handle_packaged_product(image_path, ocr_result)
        
    image_area = image.shape[0] * image.shape[1]
    segmenter = get_segmenter()
    result = segmenter.segment_image(image)
    
    if result is None or len(result.boxes) == 0:
        print("\n⚠ YOLO found no food. Triggering Gemini Zero-Shot Fallback...")
        gemini_rescue = call_gemini_vision_api(image_path, "nothing")
        if gemini_rescue and gemini_rescue.lower() != "nothing":
            print(f"   Gemini rescued the image! Detected: '{gemini_rescue}'")
            foods = [{
                "class": gemini_rescue, 
                "confidence": 0.99, 
                "pixel_area": image_area * 0.25, 
                "avg_depth": 0.5, 
                "relative_volume": (image_area * 0.25) * 0.5, 
                "is_uncertain": True, 
                "uncertain_reason": "Zero-shot rescue (No spatial segmentation)"
            }]
            dish_result = {"label": gemini_rescue, "confidence": 0.99}
            mismatch = False
            mismatch_reason = None
        else:
            return {"status": "no_food_detected", "is_packaged": False}
    else:
        depth = get_depth(image)
        foods = estimate_volume(result, depth)
        foods = merge_foods(foods)
        dish_result = classify_dish(image)
        
        # Check mismatch against original YOLO predictions BEFORE any overrides
        mismatch, mismatch_reason = check_dish_mismatch(dish_result, foods)

        # ---------------------------------------------------------------------
        # MULTI-COMPARTMENT TRAY FALLBACK
        # If YOLO only segmented 1 box on a complex/multi-item dish and the 
        # dish classifier is uncertain (< 0.40), force Gemini full-tray analysis
        # ---------------------------------------------------------------------
        dish_conf = dish_result.get("confidence", 0) if dish_result else 0
        if len(foods) <= 1 and dish_conf < 0.40:
            print("\n⚠️ Single/weak segmentation on a complex meal tray detected.")
            print("   Triggering Gemini Vision API for full tray itemization...")
            gemini_tray_items = call_gemini_vision_api(image_path, "complex_tray_breakdown")
            if gemini_tray_items:
                print(f"   Gemini detected tray components: '{gemini_tray_items}'")
                # Override the single poor box with Gemini's detailed items
                foods = [{
                    "class": item.strip(),
                    "confidence": 0.95,
                    "pixel_area": (image_area * 0.20),
                    "avg_depth": 0.5,
                    "relative_volume": (image_area * 0.20) * 0.5,
                    "is_uncertain": False,
                    "uncertain_reason": "Gemini Multi-Compartment Tray Breakdown"
                } for item in gemini_tray_items.split(",") if item.strip()]
                
                mismatch = False
                mismatch_reason = None
        
    nutrition_estimator = NutritionEstimator()
    ingredient_cards = []
    
    # Expanded Blacklist: Top dataset majority classes & known class sinks
    SUSPICIOUS_CLASSES = [
        # Explicit Dataset Sink & Imbalance Artifacts
        "chicken duck",
        "other ingredients",
        "sauce",
        "soup",
        "mixed meal",
        "unknown",
        
        # Secondary Imbalanced Meats & Sinks
        "fried meat",
        "steak",
        
        # Common Garnish / Glare False-Positives
        "lemon",
        "garlic",
        "ginger"
    ]
    
    for food in foods:
        is_uncertain, uncertain_reason = check_dominant_detection(food, image_area)
        
        yolo_class = food["class"].lower()
        yolo_conf = food["confidence"]
        dish_label = dish_result.get("label", "").lower() if dish_result else ""
        dish_conf = dish_result.get("confidence", 0) if dish_result else 0
        
        # Trigger Gemini Fallback when predictions are ambiguous, mismatched, or belong to majority sinks
        force_fallback = (
            is_uncertain 
            or mismatch 
            or yolo_class in SUSPICIOUS_CLASSES 
            or yolo_conf < 0.70 
            or dish_conf < 0.70
        )
        
        if force_fallback:
            print(f"\n🤖 Triggering Gemini API Fallback (YOLO: '{food['class']}', Dish: '{dish_label}', Conf: {yolo_conf:.2f})...")
            gemini_class = call_gemini_vision_api(image_path, food["class"])
            if gemini_class:
                print(f"   Gemini identified food as: '{gemini_class}'")
                food["class"] = gemini_class
                food["confidence"] = 0.99
                is_uncertain = False
                uncertain_reason = "Corrected by Gemini Vision API"
        elif dish_conf > 0.70 and yolo_conf < 0.65:
            # Fall back to high-confidence dish classifier if Gemini wasn't required
            print(f"🔄 OVERRIDE: '{food['class']}' -> '{dish_result['label']}'")
            food["class"] = dish_result["label"]
            food["confidence"] = float(dish_result["confidence"])
                    
        info = nutrition_estimator.get_nutrition(food["class"])
        canonical_food = info["canonical_food"] if info is not None else food["class"]
        mass_grams = float(estimate_mass(food["relative_volume"], canonical_food))
        
        card = {
            "ingredient": food["class"].title(), 
            "confidence": float(food["confidence"]), 
            "estimated_mass_g": round(mass_grams, 2), 
            "is_uncertain": is_uncertain, 
            "uncertain_reason": uncertain_reason, 
            "pixel_area": int(food["pixel_area"]), 
            "avg_depth": float(food["avg_depth"]), 
            "relative_volume": float(food["relative_volume"])
        }
        
        if info is not None:
            scaled = scale_nutrition(info["nutrition"], mass_grams)
            card["nutrition"] = {k: float(v) for k, v in scaled.items()}
        else:
            card["nutrition"] = None
            
        ingredient_cards.append(card)
        
    meal_name = infer_meal(foods)
    geometry = total_geometry(foods)
    nutrition_totals = total_nutrition(ingredient_cards)
    
    return {
        "is_packaged": False, 
        "status": "success", 
        "image": Path(image_path).name, 
        "meal_name": meal_name, 
        "dish_classifier": {
            "label": dish_result["label"] if dish_result else None, 
            "confidence": float(dish_result["confidence"]) if dish_result else None
        }, 
        "mismatch_warning": mismatch_reason if mismatch else None, 
        "geometry_totals": {
            "pixel_area": int(geometry["pixel_area"]), 
            "relative_volume": float(geometry["relative_volume"]), 
            "estimated_mass_g": round(sum(c["estimated_mass_g"] for c in ingredient_cards), 2)
        }, 
        "nutrition_totals": {k: float(v) for k, v in nutrition_totals.items()} if nutrition_totals else None, 
        "ingredients": ingredient_cards
    }
def main():

    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        image_path = ROOT / "assets" / "test.jpeg"

    output = process_image(image_path)
    import json
    print("\n========== PIPELINE RESULT JSON ==========")
    print(json.dumps(output, indent=2))
    print("==========================================\n")


if __name__ == "__main__":
    main()