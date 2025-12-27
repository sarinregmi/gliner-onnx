#!/usr/bin/env python3
"""
Final Unified Benchmark Script

This script provides a FAIR comparison between all models:
1. Single Inference (1 string)
2. Batched Inference (8 strings, reported per-item)
- Each test starts with a clean memory state (gc.collect + torch.cuda.empty_cache).
- ONNX providers are explicitly verified to ensure optimal performance.
"""

import time
import torch
import sys
import os
import warnings
import gc
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="onnxconverter_common")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*"
)

# Detect project path
PROJECT_PATH = Path(__file__).parent
sys.path.insert(0, str(PROJECT_PATH))

from gliner import GLiNER

# Check for ONNX Runtime
try:
    import onnxruntime as ort
    from onnxconverter_common import float16
    import onnx

    AVAILABLE_PROVIDERS = ort.get_available_providers()
except ImportError:
    AVAILABLE_PROVIDERS = []
    ort = None

# Hardware detection
HAS_CUDA = torch.cuda.is_available()


def clear_memory():
    """Wipe memory to prevent interference between tests."""
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_system_info():
    """Print detected system configuration."""
    print("=" * 80)
    print("SYSTEM CONFIGURATION")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available (PyTorch): {HAS_CUDA}")
    if HAS_CUDA:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"ONNX Runtime providers available: {AVAILABLE_PROVIDERS}")
    print("=" * 80)


def convert_to_fp16_if_missing(fp32_path, fp16_path):
    """Ensure FP16 model exists for testing."""
    if fp16_path.exists():
        return True

    print(f"  Converting {fp32_path.name} to FP16...")
    try:
        model = onnx.load(fp32_path)
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, fp16_path)
        print(f"  ✓ Saved to {fp16_path.name}")
        return True
    except Exception as e:
        print(f"  ❌ Conversion failed: {e}")
        return False


def benchmark_single(model, text, labels, iterations=20, use_amp=False):
    """Benchmark raw single inference (one call, one string)."""
    # Warmup
    for _ in range(3):
        if use_amp:
            with torch.amp.autocast("cuda"):
                _ = model.predict_entities(text, labels)
        else:
            _ = model.predict_entities(text, labels)

    clear_memory()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        if use_amp:
            with torch.amp.autocast("cuda"):
                _ = model.predict_entities(text, labels)
        else:
            _ = model.predict_entities(text, labels)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times)


def benchmark_batched(model, text, labels, batch_size=8, iterations=10, use_amp=False):
    """Benchmark through-put (batch of strings)."""
    batch_texts = [text] * batch_size

    # Warmup
    for _ in range(3):
        if use_amp:
            with torch.amp.autocast("cuda"):
                _ = model.inference(batch_texts, labels)
        else:
            _ = model.inference(batch_texts, labels)

    clear_memory()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        if use_amp:
            with torch.amp.autocast("cuda"):
                _ = model.inference(batch_texts, labels)
        else:
            _ = model.inference(batch_texts, labels)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_batch_time = sum(times) / len(times)
    return avg_batch_time / batch_size  # Per-item time


def run_benchmark():
    """Run fair, consolidated benchmark suite."""
    text = "My name is John Doe and I work at Microsoft. My email is john.doe@microsoft.com and my SSN is 123-45-6789."
    labels = ["person", "organization", "email", "ssn"]
    models_dir = PROJECT_PATH / "models"

    print_system_info()
    print()

    results_single = {}
    results_batch = {}

    # --- [1] PyTorch CPU (Baseline) ---
    print("Testing PyTorch CPU...")
    model = GLiNER.from_pretrained("nvidia/gliner-PII").to("cpu")
    results_single["PyTorch CPU"] = benchmark_single(model, text, labels)
    del model
    clear_memory()

    # --- [2] PyTorch CUDA FP32 ---
    if HAS_CUDA:
        print("Testing PyTorch CUDA FP32...")
        # Load directly to cuda
        model = GLiNER.from_pretrained("nvidia/gliner-PII", map_location="cuda")
        model.eval()
        results_single["PyTorch CUDA FP32"] = benchmark_single(model, text, labels)
        results_batch["PyTorch CUDA FP32"] = benchmark_batched(model, text, labels)
        del model
        clear_memory()

    # --- [3] PyTorch CUDA FP16 (AMP) ---
    if HAS_CUDA:
        print("Testing PyTorch CUDA FP16 (AMP)...")
        model = GLiNER.from_pretrained("nvidia/gliner-PII", map_location="cuda")
        model.eval()
        results_single["PyTorch CUDA FP16 (AMP)"] = benchmark_single(
            model, text, labels, use_amp=True
        )
        results_batch["PyTorch CUDA FP16 (AMP)"] = benchmark_batched(
            model, text, labels, use_amp=True
        )
        del model
        clear_memory()

    # --- [4] ONNX CUDA FP32 ---
    fp32_path = models_dir / "model.onnx"
    if fp32_path.exists() and "CUDAExecutionProvider" in AVAILABLE_PROVIDERS:
        print("Testing ONNX CUDA FP32...")
        try:
            # IMPORTANT: map_location='cuda' is required to trigger CUDAExecutionProvider in the wrapper
            model = GLiNER.from_pretrained(
                str(models_dir),
                load_onnx_model=True,
                onnx_model_file=fp32_path.name,
                map_location="cuda",
            )
            results_single["ONNX CUDA FP32"] = benchmark_single(model, text, labels)
            results_batch["ONNX CUDA FP32"] = benchmark_batched(model, text, labels)
            del model
            clear_memory()
        except Exception as e:
            print(f"  ❌ ONNX FP32 failed: {e}")

    # --- [5] ONNX CUDA FP16 ---
    fp16_path = models_dir / "model_fp16.onnx"
    if fp32_path.exists() and "CUDAExecutionProvider" in AVAILABLE_PROVIDERS:
        if convert_to_fp16_if_missing(fp32_path, fp16_path):
            print("Testing ONNX CUDA FP16...")
            try:
                # IMPORTANT: map_location='cuda' is required
                model = GLiNER.from_pretrained(
                    str(models_dir),
                    load_onnx_model=True,
                    onnx_model_file=fp16_path.name,
                    map_location="cuda",
                )
                results_single["ONNX CUDA FP16"] = benchmark_single(model, text, labels)
                results_batch["ONNX CUDA FP16"] = benchmark_batched(model, text, labels)
                del model
                clear_memory()
            except Exception as e:
                print(f"  ❌ ONNX FP16 failed: {e}")

    # --- [6] ONNX CPU Quantized ---
    quant_path = models_dir / "model_quantized.onnx"
    if quant_path.exists():
        print("Testing ONNX CPU Quantized (INT8)...")
        try:
            model = GLiNER.from_pretrained(
                str(models_dir), load_onnx_model=True, onnx_model_path=str(quant_path)
            ).to("cpu")
            results_single["ONNX CPU Quantized"] = benchmark_single(model, text, labels)
            del model
            clear_memory()
        except Exception as e:
            print(f"  ❌ Quantized failed: {e}")

    # Final Summary Table
    print("\n" + "=" * 80)
    print(f"{'CONFIGURATION':<35} | {'SINGLE (ms)':<15} | {'BATCH (per-item)'}")
    print("-" * 80)

    all_keys = sorted(
        set(list(results_single.keys()) + list(results_batch.keys())),
        key=lambda x: results_single.get(x, 9999),
    )

    for k in all_keys:
        single_str = (
            f"{results_single[k]:>8.2f} ms"
            if k in results_single
            else "      N/A      "
        )
        batch_str = (
            f"{results_batch[k]:>8.2f} ms" if k in results_batch else "      N/A      "
        )
        winner_mark = "⭐" if k == min(results_single, key=results_single.get) else "  "
        print(f"{winner_mark} {k:<32} | {single_str:<15} | {batch_str}")

    print("=" * 80)
    print("FAIR COMPARISON NOTES:")
    print(
        "1. 'SINGLE' is 1 string input, exactly how it would be used in a simple request."
    )
    print("2. 'BATCH' is the cost PER-ITEM when processing 8 strings at once.")
    print("3. Memory was cleared between every test to ensure zero interference.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
