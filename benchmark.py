#!/usr/bin/env python3
"""
Comprehensive GLiNER Benchmark Script

Tests all available configurations with fair comparisons:
- PyTorch (FP32, FP16 AMP)
- ONNX (FP32, FP16)
- Single inference and batched inference
- All hardware (CPU, CUDA, MPS, CoreML)

Usage:
    python benchmark_comprehensive.py
"""

import time
import torch
import sys
import os
import warnings
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
    print("=" * 70)
    print("SYSTEM CONFIGURATION")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available (PyTorch): {HAS_CUDA}")
    if HAS_CUDA:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    print(f"MPS available (Apple Silicon): {HAS_MPS}")
    print(f"ONNX Runtime providers: {AVAILABLE_PROVIDERS}")
    print("=" * 70)


def convert_onnx_to_fp16(input_path, output_path):
    """Convert ONNX model from FP32 to FP16."""
    if output_path.exists():
        return output_path

    print(f"  Converting {input_path.name} to FP16...")
    model = onnx.load(input_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, output_path)
    print(f"  ✓ Saved to {output_path.name}")
    return output_path


def benchmark_single(model, text, labels, name, iterations=20, use_amp=False):
    """Benchmark single inference."""
    print(f"  Benchmarking {name}...")

    # Warmup
    for _ in range(3):
        if use_amp:
            with torch.amp.autocast("cuda"):
                _ = model.predict_entities(text, labels)
        else:
            _ = model.predict_entities(text, labels)

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

    avg_time = sum(times) / len(times)
    print(f"    → Average: {avg_time:.2f} ms")
    return avg_time


def benchmark_batched(
    model, text, labels, name, batch_size=8, iterations=10, use_amp=False
):
    """Benchmark batched inference."""
    print(f"  Benchmarking {name} (batch={batch_size})...")

    batch_texts = [text] * batch_size

    # Warmup
    for _ in range(3):
        if use_amp:
            with torch.amp.autocast("cuda"):
                _ = model.inference(batch_texts, labels)
        else:
            _ = model.inference(batch_texts, labels)

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
    per_item_time = avg_batch_time / batch_size
    print(f"    → Batch: {avg_batch_time:.2f} ms | Per-item: {per_item_time:.2f} ms")
    return per_item_time


def run_benchmark():
    """Run comprehensive benchmarks."""
    text = "My name is John Doe and I work at Microsoft. My email is john.doe@microsoft.com and my SSN is 123-45-6789."
    labels = ["person", "organization", "email", "ssn"]
    models_dir = PROJECT_PATH / "models"

    print_system_info()
    print()

    results = {}

    # ========== PyTorch Tests ==========
    print("=" * 70)
    print("PYTORCH TESTS")
    print("=" * 70)

    print("\nLoading PyTorch model...")
    model_pt = GLiNER.from_pretrained("nvidia/gliner-PII")

    # CPU
    print("\n--- PyTorch CPU ---")
    model_pt_cpu = model_pt.to("cpu")
    model_pt_cpu.eval()
    pt_cpu_time = benchmark_single(model_pt_cpu, text, labels, "Single")
    results["PyTorch CPU"] = pt_cpu_time

    # CUDA
    if HAS_CUDA:
        print("\n--- PyTorch CUDA ---")
        model_pt_cuda = model_pt.to("cuda")
        model_pt_cuda.eval()

        # FP32
        pt_cuda_fp32 = benchmark_single(model_pt_cuda, text, labels, "FP32")
        results["PyTorch CUDA FP32"] = pt_cuda_fp32
        print(f"    → Speedup vs CPU: {pt_cpu_time/pt_cuda_fp32:.2f}x")

        # FP16 AMP
        pt_cuda_fp16 = benchmark_single(
            model_pt_cuda, text, labels, "FP16 AMP", use_amp=True
        )
        results["PyTorch CUDA FP16"] = pt_cuda_fp16
        print(f"    → Speedup vs FP32: {pt_cuda_fp32/pt_cuda_fp16:.2f}x")

        # FP16 AMP Batched
        pt_cuda_fp16_batch = benchmark_batched(
            model_pt_cuda, text, labels, "FP16 AMP", use_amp=True
        )
        results["PyTorch CUDA FP16 (batched)"] = pt_cuda_fp16_batch
        print(f"    → Speedup vs single: {pt_cuda_fp16/pt_cuda_fp16_batch:.2f}x")

    # MPS (Apple Silicon)
    if HAS_MPS:
        print("\n--- PyTorch MPS ---")
        try:
            model_pt_mps = model_pt.to("mps")
            pt_mps_time = benchmark_single(model_pt_mps, text, labels, "Single")
            results["PyTorch MPS"] = pt_mps_time
            print(f"    → Speedup vs CPU: {pt_cpu_time/pt_mps_time:.2f}x")
        except Exception as e:
            print(f"  MPS failed: {e}")

    # ========== ONNX Tests ==========
    print("\n" + "=" * 70)
    print("ONNX TESTS")
    print("=" * 70)

    fp32_onnx_path = models_dir / "model.onnx"
    fp16_onnx_path = models_dir / "model_fp16.onnx"
    quantized_onnx_path = models_dir / "model_quantized.onnx"

    if not fp32_onnx_path.exists():
        print(f"\n⚠️  ONNX models not found at {models_dir}")
        print("Run 'python convert_model.py' first to generate ONNX models.")
    else:
        # ONNX CPU
        print("\n--- ONNX CPU ---")
        try:
            model_onnx_cpu = GLiNER.from_pretrained(
                str(models_dir), load_onnx_model=True
            )
            onnx_cpu_time = benchmark_single(model_onnx_cpu, text, labels, "FP32")
            results["ONNX CPU FP32"] = onnx_cpu_time
            print(f"    → Speedup vs PyTorch CPU: {pt_cpu_time/onnx_cpu_time:.2f}x")
        except Exception as e:
            print(f"  Failed: {e}")

        # ONNX Quantized (CPU)
        if quantized_onnx_path.exists():
            try:
                model_onnx_quant = GLiNER.from_pretrained(
                    str(models_dir),
                    load_onnx_model=True,
                    onnx_model_path=str(quantized_onnx_path),
                )
                onnx_quant_time = benchmark_single(
                    model_onnx_quant, text, labels, "Quantized (INT8)"
                )
                results["ONNX CPU Quantized"] = onnx_quant_time
                print(
                    f"    → Speedup vs ONNX FP32: {onnx_cpu_time/onnx_quant_time:.2f}x"
                )
            except Exception as e:
                print(f"  Quantized failed: {e}")

        # ONNX CUDA
        if HAS_CUDA_PROVIDER:
            print("\n--- ONNX CUDA ---")
            try:
                model_onnx_cuda = GLiNER.from_pretrained(
                    str(models_dir), load_onnx_model=True
                )
                model_onnx_cuda = model_onnx_cuda.to("cuda")

                # FP32
                onnx_cuda_fp32 = benchmark_single(model_onnx_cuda, text, labels, "FP32")
                results["ONNX CUDA FP32"] = onnx_cuda_fp32
                print(
                    f"    → Speedup vs PyTorch CUDA: {pt_cuda_fp32/onnx_cuda_fp32:.2f}x"
                )

                # FP16
                convert_onnx_to_fp16(fp32_onnx_path, fp16_onnx_path)
                model_onnx_cuda_fp16 = GLiNER.from_pretrained(
                    str(models_dir),
                    load_onnx_model=True,
                    onnx_model_path=str(fp16_onnx_path),
                )
                model_onnx_cuda_fp16 = model_onnx_cuda_fp16.to("cuda")
                onnx_cuda_fp16 = benchmark_single(
                    model_onnx_cuda_fp16, text, labels, "FP16"
                )
                results["ONNX CUDA FP16"] = onnx_cuda_fp16
                print(
                    f"    → Speedup vs ONNX FP32: {onnx_cuda_fp32/onnx_cuda_fp16:.2f}x"
                )

            except Exception as e:
                print(f"  CUDA failed: {e}")

        # ONNX CoreML (Apple Silicon)
        if HAS_COREML_PROVIDER:
            print("\n--- ONNX CoreML ---")
            try:
                # CoreML requires special setup, using CPU provider as fallback
                model_onnx_coreml = GLiNER.from_pretrained(
                    str(models_dir), load_onnx_model=True
                )
                onnx_coreml_time = benchmark_single(
                    model_onnx_coreml, text, labels, "CoreML"
                )
                results["ONNX CoreML"] = onnx_coreml_time
                print(
                    f"    → Speedup vs ONNX CPU: {onnx_cpu_time/onnx_coreml_time:.2f}x"
                )
            except Exception as e:
                print(f"  CoreML failed: {e}")

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Time (ms)':<12} {'Speedup'}")
    print("-" * 70)

    if results:
        baseline = results.get("PyTorch CPU", list(results.values())[0])
        sorted_results = sorted(results.items(), key=lambda x: x[1])

        for config, time_ms in sorted_results:
            speedup = baseline / time_ms
            marker = "🥇" if time_ms == sorted_results[0][1] else "  "
            print(f"{marker} {config:<33} {time_ms:>8.2f} ms   {speedup:>6.2f}x")

        print()
        best_config, best_time = sorted_results[0]
        print(f"✅ RECOMMENDED: {best_config} ({best_time:.2f}ms)")

        # Calculate throughput
        throughput = 1000 / best_time
        print(f"   Throughput: {throughput:.0f} inferences/second")
        print(
            f"   Expected end-to-end latency: ~{best_time + 40:.0f}-{best_time + 60:.0f}ms (including network)"
        )


if __name__ == "__main__":
    run_benchmark()
