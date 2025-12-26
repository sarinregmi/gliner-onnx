import sys
import os
from pathlib import Path


# The gliner package is now local to this project
# so we can import it directly.


import torch

try:
    from gliner import GLiNER

    print(f"Successfully imported GLiNER from {GLiNER.__module__}")
except ImportError as e:
    print(f"Failed to import GLiNER: {e}")
    sys.exit(1)


def convert():
    model_id = "nvidia/gliner-PII"
    save_dir = Path("models")

    print(f"Loading model: {model_id}")
    # Load model using the local class
    model = GLiNER.from_pretrained(model_id, load_tokenizer=True)

    print("Model loaded. Starting ONNX export...")

    # Verify export_to_onnx exists
    if not hasattr(model, "export_to_onnx"):
        print("ERROR: This version of GLiNER does not have export_to_onnx method!")
        print(f"GLiNER file: {sys.modules['gliner'].__file__}")
        return

    try:
        output_paths = model.export_to_onnx(
            save_dir=save_dir,
            onnx_filename="model.onnx",
            quantized_filename="model_quantized.onnx",
            quantize=True,
            opset=17,  # Using 17 as a safe default, user example had 19
        )
        print("\n----------------------------------------")
        print("✓ Conversion complete!")
        print(f"ONNX Model: {output_paths['onnx_path']}")
        print(f"Quantized Model: {output_paths['quantized_path']}")
        print("----------------------------------------")

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    convert()
