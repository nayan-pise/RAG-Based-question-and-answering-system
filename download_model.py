import os
from sentence_transformers import SentenceTransformer

print("Attempting to download the AI model...")
print("This may take a minute. Please keep your internet stable.")

try:
    # This downloads the model to your local cache
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("\n✅ Success! Model downloaded.")
    print("You can now run 'python -m uvicorn main:app --reload'")
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("Try connecting to a different network (like a mobile hotspot) and try again.")