#!/usr/bin/env python3
"""
GLiNER ONNX Benchmark Script

Automatically detects available hardware (CPU, CUDA, MPS) and benchmarks
all available model variants.

Usage:
    python benchmark.py
"""

import time
import torch
import sys
import os

# Detect project path dynamically
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_PATH)

from gliner import GLiNER

# Check for ONNX Runtime providers
try:
    import onnxruntime as ort

    AVAILABLE_PROVIDERS = ort.get_available_providers()
except ImportError:
    AVAILABLE_PROVIDERS = []

# Detect hardware
HAS_CUDA = torch.cuda.is_available()
HAS_MPS = (
    getattr(torch.backends, "mps", None) is not None
    and torch.backends.mps.is_available()
)
HAS_CUDA_PROVIDER = "CUDAExecutionProvider" in AVAILABLE_PROVIDERS
HAS_COREML_PROVIDER = "CoreMLExecutionProvider" in AVAILABLE_PROVIDERS


def print_system_info():
    """Print detected system configuration."""
    print("=" * 60)
    print("SYSTEM CONFIGURATION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available (PyTorch): {HAS_CUDA}")
    if HAS_CUDA:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    print(f"MPS available (Apple Silicon): {HAS_MPS}")
    print(f"ONNX Runtime providers: {AVAILABLE_PROVIDERS}")
    print("=" * 60)


def benchmark_model(model, name, text, iterations=20):
    """Benchmark a model and return average inference time."""
    print(f"Benchmarking {name}...")

    # Warmup
    for _ in range(3):
        model.predict_entities(text, labels=["person"])

    start_time = time.time()
    for _ in range(iterations):
        model.predict_entities(text, labels=["person"])
    end_time = time.time()

    avg_ms = ((end_time - start_time) / iterations) * 1000
    print(f"  -> Average time per inference: {avg_ms:.2f} ms")
    return avg_ms


def run_benchmark():
    """Run benchmarks on all available configurations."""
    text = "My name is John Doe and I work at Microsoft in Seattle. This is a longer sentence to make sure we are testing the model's performance on something realistic."
    models_dir = os.path.join(PROJECT_PATH, "models")

    print_system_info()
    results = {}

    # 1. PyTorch CPU (baseline)
    print("\n--- PyTorch (CPU) ---")
    model_pt = GLiNER.from_pretrained("nvidia/gliner-PII", load_tokenizer=True).to(
        "cpu"
    )
    model_pt.eval()
    pt_cpu_time = benchmark_model(model_pt, "PyTorch CPU", text)
    results["PyTorch CPU"] = pt_cpu_time

    # 2. PyTorch CUDA (if available)
    if HAS_CUDA:
        print("\n--- PyTorch (CUDA) ---")
        try:
            model_pt_cuda = model_pt.to("cuda")
            pt_cuda_time = benchmark_model(model_pt_cuda, "PyTorch CUDA", text)
            results["PyTorch CUDA"] = pt_cuda_time
            print(f"  -> Speedup vs CPU: {pt_cpu_time/pt_cuda_time:.2f}x")
        except Exception as e:
            print(f"PyTorch CUDA failed: {e}")

    # 3. PyTorch MPS (if available - Apple Silicon)
    if HAS_MPS:
        print("\n--- PyTorch (MPS) ---")
        try:
            model_pt_mps = model_pt.to("mps")
            pt_mps_time = benchmark_model(model_pt_mps, "PyTorch MPS", text)
            results["PyTorch MPS"] = pt_mps_time
            print(f"  -> Speedup vs CPU: {pt_cpu_time/pt_mps_time:.2f}x")
        except Exception as e:
            print(f"PyTorch MPS failed: {e}")

    # 4. ONNX Standard (CPU)
    print("\n--- ONNX Standard (CPU) ---")
    try:
        model_onnx = GLiNER.from_pretrained(
            models_dir, load_onnx_model=True, onnx_model_file="model.onnx"
        )
        onnx_cpu_time = benchmark_model(model_onnx, "ONNX Standard (CPU)", text)
        results["ONNX Standard CPU"] = onnx_cpu_time
        print(f"  -> Speedup vs PyTorch CPU: {pt_cpu_time/onnx_cpu_time:.2f}x")
    except Exception as e:
        print(f"Failed to load ONNX Standard: {e}")

    # 5. ONNX Quantized (CPU) - Best for CPU
    print("\n--- ONNX Quantized (CPU) ---")
    try:
        model_q = GLiNER.from_pretrained(
            models_dir, load_onnx_model=True, onnx_model_file="model_quantized.onnx"
        )
        onnx_q_time = benchmark_model(model_q, "ONNX Quantized (CPU)", text)
        results["ONNX Quantized CPU"] = onnx_q_time
        print(f"  -> Speedup vs PyTorch CPU: {pt_cpu_time/onnx_q_time:.2f}x")
    except Exception as e:
        print(f"Failed to load ONNX Quantized: {e}")

    # 6. ONNX Standard (CUDA) - if available
    if HAS_CUDA_PROVIDER:
        print("\n--- ONNX Standard (CUDA) ---")
        try:
            model_onnx_cuda = GLiNER.from_pretrained(
                models_dir,
                load_onnx_model=True,
                onnx_model_file="model.onnx",
                map_location="cuda",
            )
            onnx_cuda_time = benchmark_model(
                model_onnx_cuda, "ONNX Standard (CUDA)", text
            )
            results["ONNX Standard CUDA"] = onnx_cuda_time
            print(f"  -> Speedup vs PyTorch CPU: {pt_cpu_time/onnx_cuda_time:.2f}x")
        except Exception as e:
            print(f"Failed to load ONNX Standard (CUDA): {e}")

    # 7. ONNX Standard (CoreML/MPS) - if available (Mac)
    if HAS_COREML_PROVIDER:
        print("\n--- ONNX Standard (CoreML/MPS) ---")
        try:
            model_onnx_mps = GLiNER.from_pretrained(
                models_dir,
                load_onnx_model=True,
                onnx_model_file="model.onnx",
                map_location="mps",
            )
            onnx_mps_time = benchmark_model(
                model_onnx_mps, "ONNX Standard (CoreML)", text
            )
            results["ONNX Standard CoreML"] = onnx_mps_time
            print(f"  -> Speedup vs PyTorch CPU: {pt_cpu_time/onnx_mps_time:.2f}x")
        except Exception as e:
            print(f"Failed to load ONNX Standard (CoreML): {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<30} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    for name, time_ms in sorted(results.items(), key=lambda x: x[1]):
        speedup = pt_cpu_time / time_ms
        print(f"{name:<30} {time_ms:<15.2f} {speedup:<10.2f}x")
    print("=" * 60)

    # Recommend best option
    if results:
        best = min(results, key=results.get)
        print(f"\n✅ RECOMMENDED: {best} ({results[best]:.2f}ms)")


if __name__ == "__main__":
    run_benchmark()
