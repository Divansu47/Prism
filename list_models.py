import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

print("Fetching available models from Gemini API...\n")

try:
    # Use pagination to grab the model names directly
    for m in client.models.list():
        print(f"- {m.name}")
except Exception as e:
    print(f"Failed to fetch models: {e}")