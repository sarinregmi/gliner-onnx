"""
SLM (Small Language Model) Benchmark Script
Tests inference latency for Phi-4-mini and similar models using vLLM.
This complements the GLiNER benchmark to validate the 3-layer hybrid architecture.
"""

import time
import argparse
import json
import torch
import gc


def force_cleanup():
    """Force garbage collection and GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_vllm(model_name: str, num_runs: int = 20):
    """Benchmark vLLM inference latency."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        return {"error": "vLLM not installed. Run: pip install vllm"}

    print(f"\n--- Benchmarking {model_name} with vLLM ---")

    # Test prompts (similar to SLM detection prompts)
    test_prompts = [
        """Analyze the following text for attorney-client privileged communication.
        
Text: "Dear Mr. Johnson, as your legal counsel, I advise you to..."

Is this privileged? Respond with JSON: {"is_privileged": true/false, "confidence": 0.0-1.0}""",
        """Analyze the following text for potential trade secrets.
        
Text: "Our proprietary algorithm uses a combination of RSA encryption with..."

Contains trade secret? Respond with JSON: {"contains_trade_secret": true/false, "confidence": 0.0-1.0}""",
    ]

    # Initialize model
    print(f"Loading model: {model_name}")
    load_start = time.perf_counter()

    try:
        llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.5,  # Leave room for GLiNER
            max_model_len=2048,
            trust_remote_code=True,
        )
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    load_time = (time.perf_counter() - load_start) * 1000
    print(f"Model loaded in {load_time:.2f}ms")

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=128,
        stop=["```", "\n\n\n"],
    )

    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = llm.generate(test_prompts[:1], sampling_params)

    # Benchmark single inference
    print(f"Running {num_runs} single inference tests...")
    single_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        outputs = llm.generate([test_prompts[0]], sampling_params)
        end = time.perf_counter()
        single_times.append((end - start) * 1000)

    avg_single = sum(single_times) / len(single_times)

    # Benchmark batch inference
    print(f"Running {num_runs} batch inference tests (batch_size=4)...")
    batch_prompts = test_prompts * 2  # 4 prompts
    batch_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        outputs = llm.generate(batch_prompts, sampling_params)
        end = time.perf_counter()
        batch_times.append((end - start) * 1000)

    avg_batch = sum(batch_times) / len(batch_times)
    avg_per_item = avg_batch / len(batch_prompts)

    # Get GPU memory usage
    gpu_memory_used = 0
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB

    return {
        "model": model_name,
        "load_time_ms": round(load_time, 2),
        "single_inference_ms": round(avg_single, 2),
        "batch_4_total_ms": round(avg_batch, 2),
        "batch_4_per_item_ms": round(avg_per_item, 2),
        "gpu_memory_gb": round(gpu_memory_used, 2),
    }


def benchmark_llama_cpp(model_path: str, num_runs: int = 20):
    """Benchmark llama-cpp-python inference latency (for quantized models)."""
    try:
        from llama_cpp import Llama
    except ImportError:
        return {
            "error": "llama-cpp-python not installed. Run: pip install llama-cpp-python"
        }

    print(f"\n--- Benchmarking {model_path} with llama.cpp ---")

    test_prompt = """Analyze this text for PII: "John Doe, SSN 123-45-6789"
Respond with JSON: {"contains_pii": true/false, "entities": [...]}"""

    print(f"Loading model: {model_path}")
    load_start = time.perf_counter()

    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1,  # Use all GPU layers
            verbose=False,
        )
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    load_time = (time.perf_counter() - load_start) * 1000
    print(f"Model loaded in {load_time:.2f}ms")

    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = llm(test_prompt, max_tokens=64, temperature=0.1)

    # Benchmark
    print(f"Running {num_runs} inference tests...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        output = llm(test_prompt, max_tokens=64, temperature=0.1)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = sum(times) / len(times)

    return {
        "model": model_path,
        "load_time_ms": round(load_time, 2),
        "avg_inference_ms": round(avg_time, 2),
        "min_inference_ms": round(min(times), 2),
        "max_inference_ms": round(max(times), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="SLM Benchmark Script")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "llama_cpp"],
        default="vllm",
        help="Inference backend",
    )
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark runs")
    args = parser.parse_args()

    print("=" * 70)
    print("SLM BENCHMARK (Layer 3 - Semantic Analysis)")
    print("=" * 70)

    # System info
    print(f"\nSystem Information:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        )

    # Run benchmark
    if args.backend == "vllm":
        results = benchmark_vllm(args.model, args.runs)
    else:
        results = benchmark_llama_cpp(args.model, args.runs)

    # Print results
    print("\n" + "=" * 70)
    print("SLM BENCHMARK RESULTS")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    # Cleanup
    force_cleanup()

    return results


if __name__ == "__main__":
    main()
