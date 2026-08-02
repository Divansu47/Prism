# 🌈 Prism — AI-Powered Computer Vision & Nutrition Inference Engine

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Google Cloud Run](https://img.shields.io/badge/GCP-Cloud_Run-4285F4.svg)](https://cloud.google.com/run)

**Prism** is a production-grade Computer Vision and Nutrition Analytics API. It combines state-of-the-art instance segmentation, depth-based volumetric estimation, and neural dish classification to deliver precise macro and micro-nutrient breakdowns from simple meal photographs.

---

## 🏗️ Architecture & ML Pipeline

Prism executes a multi-stage inference pipeline on every incoming image request:

```text
[ User Image ]
      │
      ├──>  1. YOLO Segmentation (foodseg_best.pt) ──> Mask & Area Extraction
      ├──>  2. Depth & Geometry Engine         ──> Volume & Mass Estimation (g)
      ├──>  3. Food-101 Classifier             ──> Plate-level Categorization
      │
      └──>  4. USDA Mapping Engine             ──> Complete Nutrients (Macros & Micros)
                  │
                  └──> [ JSON Payload Response ]


```
Key Technical HighlightsLazy-Loaded Models: Models are loaded lazily to eliminate weight reloading overhead across FastAPI requests.Volumetric Mass Estimation: Calculates relative depth and bounding geometry to estimate food mass in grams.Granular Micronutrients: Provides detailed vitamin, mineral, lipid, and caloric metrics scaled to estimated mass.🚀 Quick Start1. PrerequisitesPython 3.10+Virtual Environment (.venv)2. InstallationBash# Clone the repository
git clone https://github.com/Divansu47/Prism.git
cd Prism

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
---

3. Local Development ServerBashuvicorn main:app --reload
The API server will run locally at [http://127.0.0.1:8000](http://127.0.0.1:8000).
---
   ## 📡 API Reference

### `POST /analyze-food`

Analyze a food photograph to extract volumetric metrics and full nutritional values.

#### Request (Multipart Form-Data)

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `file` | `File (Binary)` | JPEG or PNG image of the meal |

#### Example `cURL` Request

```bash
curl -X POST "https://food-nutrition-920479426497.asia-south1.run.app/analyze-food" \
  -F "file=@/path/to/meal_image.jpeg"
```
---

☁️ DeploymentPrism is optimized for containerized deployments on Google Cloud Run.Bashgcloud run deploy food-nutrition \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --cpu-boost
📜 LicenseDistributed under the Apache License 2.0. See LICENSE for more details.
---

### How to update it in VS Code:
1. Open `C:\Games\food-nutition-ml\README.md`.
2. Delete everything inside it and paste the raw code block above.
3. Save the file (`Ctrl + S`).
4. In your terminal, run:
   ```powershell
   git add README.md
   git commit -m "docs: fix markdown formatting and code blocks in README"
   git push origin main
   ```
   ---
