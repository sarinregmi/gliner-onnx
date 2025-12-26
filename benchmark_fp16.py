#!/usr/bin/env python3
"""
Benchmark script to test FP16 mixed precision performance on GPU.
This tests PyTorch's automatic mixed precision (AMP) which should work
even though manual FP16 ONNX conversion failed.
"""

import sys
import time
import torch
from pathlib import Path

# Add local gliner to path
sys.path.insert(0, str(Path(__file__).parent / "gliner"))

from gliner import GLiNER


def benchmark_inference(model, text, labels, num_runs=10, use_amp=False):
    """Benchmark inference with optional mixed precision."""
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
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = sum(times) / len(times)
    return avg_time, entities


def main():
    print("=" * 60)
    print("FP16 MIXED PRECISION BENCHMARK")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires a GPU.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print()

    # Sample text and labels
    text = "My name is John Doe and I work at Microsoft. My email is john.doe@microsoft.com and my SSN is 123-45-6789."
    labels = ["person", "organization", "email", "ssn"]

    print("Loading model...")
    model = GLiNER.from_pretrained("nvidia/gliner-PII")
    model = model.to("cuda")
    model.eval()
    print("✓ Model loaded on GPU\n")

    # Test 1: FP32 (baseline)
    print("--- Test 1: FP32 (Standard Precision) ---")
    fp32_time, fp32_entities = benchmark_inference(
        model, text, labels, num_runs=20, use_amp=False
    )
    print(f"Average time: {fp32_time:.2f} ms")
    print(f"Entities found: {len(fp32_entities)}")
    print()

    # Test 2: FP16 Mixed Precision
    print("--- Test 2: FP16 (Mixed Precision) ---")
    fp16_time, fp16_entities = benchmark_inference(
        model, text, labels, num_runs=20, use_amp=True
    )
    print(f"Average time: {fp16_time:.2f} ms")
    print(f"Entities found: {len(fp16_entities)}")
    print()

    # Test 3: Batched FP16 (batch size 8)
    print("--- Test 3: FP16 Batched (batch=8) ---")
    batch_texts = [text] * 8
    batch_times = []
    for _ in range(10):
        start = time.perf_counter()
        with torch.cuda.amp.autocast():
            batch_entities = model.predict_entities(batch_texts, labels)
        end = time.perf_counter()
        batch_times.append((end - start) * 1000)

    avg_batch_time = sum(batch_times) / len(batch_times)
    per_item_time = avg_batch_time / 8
    print(f"Average batch time: {avg_batch_time:.2f} ms")
    print(f"Per-item time: {per_item_time:.2f} ms")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    speedup = fp32_time / fp16_time
    batch_speedup = fp32_time / per_item_time

    print(f"FP32 baseline:        {fp32_time:.2f} ms")
    print(f"FP16 mixed precision: {fp16_time:.2f} ms ({speedup:.2f}x faster)")
    print(
        f"FP16 batched (per-item): {per_item_time:.2f} ms ({batch_speedup:.2f}x faster)"
    )
    print()

    if speedup > 1.2:
        print(f"✅ FP16 provides {speedup:.1f}x speedup - RECOMMENDED for production!")
    elif speedup > 1.05:
        print(f"⚠️  FP16 provides modest {speedup:.1f}x speedup - may be worth it")
    else:
        print(f"❌ FP16 provides minimal speedup - stick with FP32")

    print()
    print(f"💡 With batching, you can achieve ~{per_item_time:.1f}ms per inference!")


if __name__ == "__main__":
    main()
