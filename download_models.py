import os
import gdown

os.makedirs("weights", exist_ok=True)

FILE_ID = "12FbLEMR1UsMYJwPquTDjIumN_bBIzDjX"

url = f"https://drive.google.com/uc?id={FILE_ID}"

output = "weights/segmentation_model.pth"

if not os.path.exists(output):
    print("Downloading segmentation model...")
    gdown.download(url, output, quiet=False)

if not os.path.exists(output):
    raise RuntimeError("Model download failed!")

print("Model ready.")