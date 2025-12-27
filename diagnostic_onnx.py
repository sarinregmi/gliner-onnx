#!/usr/bin/env python3
"""
Diagnostic script to test raw ONNX performance vs GLiNER wrapper.
Also attempts a more robust FP16 conversion.
"""

import time
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import onnx
import warnings

# Detect project path
PROJECT_PATH = Path(__file__).parent
sys.path.insert(0, str(PROJECT_PATH))

from gliner import GLiNER


def robust_convert_to_fp16(input_path, output_path):
    """Try a more robust FP16 conversion for BERT models."""
    from onnxconverter_common import float16

    print(f"Loading model from {input_path}...")
    model = onnx.load(input_path)

    print("Converting to FP16 with robust settings...")
    try:
        # Some nodes in BERT are sensitive to FP16, we try to keep them as float
        # We also enable 'keep_io_types' which is crucial for some providers
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            disable_fast_math=True,  # Can help with precision issues in some ops
        )
        onnx.save(model_fp16, output_path)
        print(f"✓ Saved FP16 model to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Robust conversion failed: {e}")
        return False


def diagnostic_run():
    models_dir = PROJECT_PATH / "models"
    fp32_path = models_dir / "model.onnx"
    fp16_path = models_dir / "model_fp16_robust.onnx"

    if not fp32_path.exists():
        print(f"FP32 model missing at {fp32_path}")
        return

    text = "John Doe works at Microsoft."
    labels = ["person", "organization"]

    # 1. Test Raw ONNX Runtime (Direct Session)
    print("\n--- [1] Raw ONNX Runtime (CUDA) ---")
    session = ort.InferenceSession(str(fp32_path), providers=["CUDAExecutionProvider"])

    # Map ORT types to numpy types
    type_map = {
        "tensor(int64)": np.int64,
        "tensor(float)": np.float32,
        "tensor(bool)": np.bool_,
        "tensor(float16)": np.float16,
    }

    dummy_inputs = {}
    print("Inference Session Inputs:")
    for inp in session.get_inputs():
        dtype = type_map.get(inp.type, np.int64)
        print(f"  - {inp.name}: {inp.type} -> {dtype}")

        # Handle dynamic shapes (usually first dim is batch, often marked as None or string)
        shape = []
        for dim in inp.shape:
            if isinstance(dim, str) or dim is None:
                shape.append(1)
            else:
                shape.append(dim)

        # Special case: sequence lengths or other specifics
        if "len" in inp.name or "idx" in inp.name:
            dummy_inputs[inp.name] = np.zeros(shape, dtype=dtype)
        else:
            dummy_inputs[inp.name] = np.zeros(shape, dtype=dtype)

    # Warmup
    for _ in range(5):
        _ = session.run(None, dummy_inputs)

    times = []
    for _ in range(20):
        start = time.perf_counter()
        _ = session.run(None, dummy_inputs)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    print(f"Raw Session.run time: {sum(times)/len(times):.2f} ms")

    # 2. Test GLiNER Wrapper (ONNX)
    print("\n--- [2] GLiNER Wrapper (ONNX CUDA) ---")
    try:
        model = GLiNER.from_pretrained(
            str(models_dir), load_onnx_model=True, map_location="cuda"
        )

        # Warmup
        for _ in range(5):
            _ = model.predict_entities(text, labels)

        times = []
        for _ in range(20):
            start = time.perf_counter()
            _ = model.predict_entities(text, labels)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        print(f"Wrapper predict_entities time: {sum(times)/len(times):.2f} ms")
    except Exception as e:
        print(f"Wrapper failed: {e}")

    # 3. Attempt Robust FP16
    print("\n--- [3] Attempting Robust FP16 Conversion ---")
    if robust_convert_to_fp16(fp32_path, fp16_path):
        try:
            print("Testing robust FP16 model loading...")
            session_fp16 = ort.InferenceSession(
                str(fp16_path), providers=["CUDAExecutionProvider"]
            )
            print("✓ Successfully loaded FP16 model on CUDA!")
        except Exception as e:
            print(f"❌ Loading robust FP16 failed: {e}")


if __name__ == "__main__":
    diagnostic_run()
