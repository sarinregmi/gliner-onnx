#!/usr/bin/env python3
"""
Benchmark script to test FP16 performance on GPU.
Tests both PyTorch mixed precision and ONNX FP16 conversion.
"""

import sys
import time
import torch
import onnx
import onnxruntime as ort
from pathlib import Path
from onnx import numpy_helper

# Add local gliner to path
sys.path.insert(0, str(Path(__file__).parent / "gliner"))

from gliner import GLiNER


def convert_onnx_to_fp16(input_path, output_path):
    """Convert ONNX model from FP32 to FP16."""
    from onnxconverter_common import float16

    print(f"Converting {input_path} to FP16...")
    model = onnx.load(input_path)

    # Convert to FP16
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

    # Save
    onnx.save(model_fp16, output_path)
    print(f"✓ Saved FP16 model to {output_path}")
    return output_path


def benchmark_pytorch(model, text, labels, num_runs=10, use_amp=False):
    """Benchmark PyTorch inference with optional mixed precision."""
    times = []

    # Warmup
    for _ in range(3):
        if use_amp:
            with torch.cuda.amp.autocast():
                _ = model.predict_entities(text, labels)
        else:
            _ = model.predict_entities(text, labels)

    # Benchmark
    for _ in range(num_runs):
        start = time.perf_counter()
        if use_amp:
            with torch.cuda.amp.autocast():
                entities = model.predict_entities(text, labels)
        else:
            entities = model.predict_entities(text, labels)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = sum(times) / len(times)
    return avg_time, entities


def benchmark_onnx(model_path, text, labels, num_runs=10, use_cuda=True):
    """Benchmark ONNX model inference."""
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda
        else ["CPUExecutionProvider"]
    )

    # Load GLiNER with ONNX
    model = GLiNER.from_pretrained(
        str(model_path.parent), load_onnx_model=True, onnx_model_path=model_path
    )
    if use_cuda:
        model = model.to("cuda")

    times = []

    # Warmup
    for _ in range(3):
        _ = model.predict_entities(text, labels)

    # Benchmark
    for _ in range(num_runs):
        start = time.perf_counter()
        entities = model.predict_entities(text, labels)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = sum(times) / len(times)
    return avg_time, entities


def main():
    print("=" * 60)
    print("FP16 COMPREHENSIVE BENCHMARK")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires a GPU.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"ONNX Runtime providers: {ort.get_available_providers()}")
    print()

    # Sample text and labels
    text = "My name is John Doe and I work at Microsoft. My email is john.doe@microsoft.com and my SSN is 123-45-6789."
    labels = ["person", "organization", "email", "ssn"]

    results = {}

    # ========== PyTorch Tests ==========
    print("Loading PyTorch model...")
    pytorch_model = GLiNER.from_pretrained("nvidia/gliner-PII")
    pytorch_model = pytorch_model.to("cuda")
    pytorch_model.eval()
    print("✓ Model loaded on GPU\n")

    print("--- Test 1: PyTorch FP32 (Baseline) ---")
    fp32_time, _ = benchmark_pytorch(
        pytorch_model, text, labels, num_runs=20, use_amp=False
    )
    print(f"Average time: {fp32_time:.2f} ms\n")
    results["PyTorch FP32"] = fp32_time

    print("--- Test 2: PyTorch FP16 Mixed Precision ---")
    fp16_time, _ = benchmark_pytorch(
        pytorch_model, text, labels, num_runs=20, use_amp=True
    )
    print(f"Average time: {fp16_time:.2f} ms")
    print(f"Speedup vs FP32: {fp32_time/fp16_time:.2f}x\n")
    results["PyTorch FP16 AMP"] = fp16_time

    # ========== ONNX Tests ==========
    models_dir = Path("models")
    fp32_onnx_path = models_dir / "model.onnx"
    fp16_onnx_path = models_dir / "model_fp16.onnx"

    if fp32_onnx_path.exists():
        print("--- Test 3: ONNX FP32 CUDA ---")
        try:
            onnx_fp32_time, _ = benchmark_onnx(
                fp32_onnx_path, text, labels, num_runs=20, use_cuda=True
            )
            print(f"Average time: {onnx_fp32_time:.2f} ms")
            print(f"Speedup vs PyTorch FP32: {fp32_time/onnx_fp32_time:.2f}x\n")
            results["ONNX FP32 CUDA"] = onnx_fp32_time
        except Exception as e:
            print(f"Failed: {e}\n")

        # Convert to FP16
        print("--- Test 4: ONNX FP16 CUDA (Converting...) ---")
        try:
            convert_onnx_to_fp16(fp32_onnx_path, fp16_onnx_path)

            onnx_fp16_time, _ = benchmark_onnx(
                fp16_onnx_path, text, labels, num_runs=20, use_cuda=True
            )
            print(f"Average time: {onnx_fp16_time:.2f} ms")
            print(f"Speedup vs ONNX FP32: {onnx_fp32_time/onnx_fp16_time:.2f}x")
            print(f"Speedup vs PyTorch FP32: {fp32_time/onnx_fp16_time:.2f}x\n")
            results["ONNX FP16 CUDA"] = onnx_fp16_time
        except Exception as e:
            print(f"FP16 conversion failed: {e}\n")
    else:
        print(f"⚠️  ONNX model not found at {fp32_onnx_path}")
        print("Run 'python convert_model.py' first to generate ONNX models.\n")

    # ========== Batching Test ==========
    print("--- Test 5: Best Config with Batching (batch=8) ---")
    batch_texts = [text] * 8
    batch_times = []

    for _ in range(10):
        start = time.perf_counter()
        with torch.cuda.amp.autocast():
            _ = pytorch_model.predict_entities(batch_texts, labels)
        end = time.perf_counter()
        batch_times.append((end - start) * 1000)

    avg_batch_time = sum(batch_times) / len(batch_times)
    per_item_time = avg_batch_time / 8
    print(f"Batch time: {avg_batch_time:.2f} ms")
    print(f"Per-item time: {per_item_time:.2f} ms")
    print(f"Throughput: {1000/per_item_time:.0f} inferences/second\n")
    results["PyTorch FP16 AMP (batched)"] = per_item_time

    # ========== Summary ==========
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<30} {'Time (ms)':<12} {'Speedup'}")
    print("-" * 60)

    baseline = results.get("PyTorch FP32", fp32_time)
    sorted_results = sorted(results.items(), key=lambda x: x[1])

    for config, time_ms in sorted_results:
        speedup = baseline / time_ms
        marker = "🥇" if time_ms == sorted_results[0][1] else "  "
        print(f"{marker} {config:<28} {time_ms:>8.2f} ms   {speedup:>6.2f}x")

    print()
    best_config, best_time = sorted_results[0]
    print(f"✅ RECOMMENDED: {best_config} ({best_time:.2f}ms)")
    print(
        f"   Expected end-to-end latency: ~{best_time + 40:.0f}-{best_time + 60:.0f}ms (including network)"
    )


if __name__ == "__main__":
    main()
