# Volume Prediction Strategy for Food Nutrition

## Problem

Currently, the pipeline detects food categories but **does not estimate portion sizes**. 

Example:
- Image contains: 200g rice, 150g curry, 100g vegetable
- Current system returns: nutrition for 1 unit of each (assumes 100g)
- Actual nutrition should be scaled: 2x rice, 1.5x curry, 1x vegetable

## Root Cause

The segmentation model outputs pixel masks (which pixels belong to which food), but **doesn't convert pixels to grams**.

## Volume Estimation Approaches

### Approach 1: Pixel-to-Weight Regression (Recommended)
**Complexity:** Medium | **Accuracy:** High | **Data needed:** 200-500 images with ground-truth weights

**How it works:**
1. For each detected region, calculate:
   - Number of pixels in the mask
   - Area in image space (pixels)
2. Train a regression model: pixels → grams
3. Model learns density/thickness per category

**Training data required:**
```csv
image_id,category,pixel_area,actual_weight_g,actual_volume_ml
img001,rice,4500,180,150
img001,curry,2800,120,120
img001,vegetable,3200,100,120
```

**Pros:**
- Simple to implement
- No special hardware needed
- Works with standard camera images

**Cons:**
- Requires ground-truth weight annotations
- Assumes similar camera angles and lighting
- Works per-category (rice density ≠ curry density)

---

### Approach 2: 3D Depth Estimation
**Complexity:** High | **Accuracy:** Very High | **Data needed:** RGBD images or stereo

**How it works:**
1. Use monocular depth estimation (e.g., MiDaS)
2. Combine depth + segmentation masks to get volume
3. Estimate weight from volume + assumed density

**Training data required:**
```
- RGB images with segmentation masks
- Corresponding depth maps (from RGB-D camera or stereo rig)
- Optional: calibration for known reference objects
```

**Pros:**
- More accurate 3D reconstruction
- Handles packing/density automatically
- Generalizes to different food packing

**Cons:**
- Requires depth information (special camera)
- Much slower inference
- Complex post-processing

---

### Approach 3: Reference Object Detection
**Complexity:** Medium | **Accuracy:** Medium | **Data needed:** 100-200 images

**How it works:**
1. Detect reference object in image (plate, fork, coin, hand)
2. Use known dimensions of reference to calibrate scale
3. Convert pixel measurements to real-world cm/mm
4. Estimate volume from shape assumptions
5. Convert volume to weight using food density tables

**Training data required:**
```
- Images with known reference object
- Segmentation masks for food + reference
- Reference object metadata (actual size in cm)
```

**Pros:**
- Minimal training data
- No special hardware needed
- Robust to camera angles

**Cons:**
- Depends on reference object visibility
- Requires manual calibration per food
- Assumptions about food shape/density

---

## Recommended Implementation Path

### Phase 1: Quick Win (Week 1)
Use **plate diameter as reference** (most plates are 25cm):
```python
def estimate_volume(mask, plate_diameter_cm=25):
    # Assume plate is center of image
    # Measure radius in pixels
    # Calculate pixels_per_cm scale
    # Apply to food regions
    pass
```

### Phase 2: Proper Solution (Week 2-3)
Implement **Approach 1 (Pixel-to-Weight Regression)**:
1. Collect 200-300 real food bowl images
2. Weigh each component (rice, curry, vegetable, protein)
3. Manually annotate pixel areas
4. Train regression model: category_pixels → weight_grams
5. Integrate into inference pipeline

### Phase 3: Advanced (If needed)
Integrate **monocular depth estimation** for more robustness.

---

## Integration with Current Pipeline

### Current Inference Flow:
```python
result = pipeline.predict(image_path)
# Returns:
{
  "items": [
    {"class": "rice", "confidence": 0.85},
    {"class": "curry", "confidence": 0.92},
  ],
  "estimated_nutrition": {...}  # Assumes 100g each
}
```

### With Volume Prediction:
```python
result = pipeline.predict(image_path, estimate_volume=True)
# Returns:
{
  "items": [
    {"class": "rice", "confidence": 0.85, "weight_g": 180},
    {"class": "curry", "confidence": 0.92, "weight_g": 150},
  ],
  "estimated_nutrition": {...}  # Scaled by actual weights
}
```

### Code Changes:
1. Add `estimate_volume()` function to `SegmentationNutritionPipeline`
2. Scale nutrition profile by weight_g / 100
3. Return weight_g in `items` list
4. Update response schema in API

---

## Example: Weight Regression Model

```python
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class FoodWeightRegressor(nn.Module):
    def __init__(self, num_categories=4):
        super().__init__()
        self.category_embeddings = nn.Embedding(num_categories, 16)
        self.mlp = nn.Sequential(
            nn.Linear(17, 64),  # pixel_area + category_embedding
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # output: weight in grams
        )
    
    def forward(self, pixel_area, category_idx):
        # pixel_area: [batch_size]
        # category_idx: [batch_size]
        cat_emb = self.category_embeddings(category_idx)  # [batch, 16]
        x = torch.cat([pixel_area.unsqueeze(1), cat_emb], dim=1)  # [batch, 17]
        weight = self.mlp(x)
        return weight  # [batch, 1]
```

---

## Data Collection Checklist

- [ ] 50 rice-only images with weights
- [ ] 50 curry-only images with weights
- [ ] 50 vegetable-only images with weights
- [ ] 50 protein-only images with weights
- [ ] 100 mixed bowls with all components and weights
- [ ] Masks for each image
- [ ] CSV with: `image_id, category, pixel_area, actual_weight_g`

---

## Testing Checklist

- [ ] Model predicts reasonable weights (100-300g per component)
- [ ] Nutrition scales correctly (2x portions → 2x nutrition)
- [ ] API returns weight_g in items list
- [ ] Response validates against schema
- [ ] End-to-end test with real bowl image

---

## Performance Impact

| Approach | Inference Time | Accuracy | Data Cost |
|----------|---|---|---|
| Current (no volume) | 1-2s | Low | None |
| Pixel-to-weight | 1-2s (same) | High | Medium (200-300 images) |
| Depth estimation | 5-10s | Very High | High (RGBD camera) |
| Reference object | 2-3s | Medium | Low (100 images) |

**Recommendation:** Start with Pixel-to-Weight Regression (Approach 1).

---

## Next Steps

1. Schedule time to collect labeled food bowl images
2. Set up annotation process (weight each component)
3. Train pixel-area-to-weight model
4. Integrate into `SegmentationNutritionPipeline.predict()`
5. Update API response schema
6. Retrain on real data for 10+ epochs
7. Validate accuracy on held-out test set

See `src/segmentation_pipeline/volume_prediction.py` for implementation skeleton (if added).
