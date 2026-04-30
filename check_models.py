import os
from google import genai

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "convertionalai-d8da9e4d43dd.json"

PROJECT_ID = "convertionalai"
LOCATION = "us-central1"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

print(f"--- Models in {PROJECT_ID} ({LOCATION}) ---")
try:
    for model in client.models.list():
        print(f"ID: {model.name}")
except Exception as e:
    print(f"Error listing models: {e}")
