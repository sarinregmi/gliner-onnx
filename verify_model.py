import sys
import os
from pathlib import Path

# The gliner package is now local to this project

from gliner import GLiNER


def verify():
    model_path = "models"  # Directory containing model.onnx
    print(f"Loading ONNX model from {model_path}...")

    try:
        model = GLiNER.from_pretrained(model_path, load_onnx_model=True)
        print("Model loaded successfully.")

        text = "My name is John Doe and I work at Microsoft."
        labels = ["person", "organization"]

        print(f"Predicting on text: '{text}'")
        entities = model.predict_entities(text, labels)

        print("Entities found:")
        for entity in entities:
            print(f"- {entity['text']} ({entity['label']})")

        print("\n✓ Verification successful!")

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    verify()
