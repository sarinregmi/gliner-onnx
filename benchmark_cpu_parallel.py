"""
GLiNER CPU Parallel Benchmark Script
Tests ONNX inference on CPU with parallel chunk processing.
"""

import time
import argparse
import json
import gc
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Tuple
import multiprocessing as mp

# Set thread counts before importing torch/onnx
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def get_system_info() -> dict:
    """Get CPU and system information."""
    import torch

    return {
        "cpu_count": mp.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "pytorch_version": torch.__version__,
        "python_version": os.sys.version.split()[0],
    }


def chunk_text(text: str, max_tokens: int = 450, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    Uses simple word-based chunking (not actual tokenization for speed).
    """
    words = text.split()
    chunks = []

    # Approximate tokens as words (rough estimate)
    words_per_chunk = max_tokens
    overlap_words = overlap

    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += words_per_chunk - overlap_words

        if end >= len(words):
            break

    return chunks


def run_single_inference_pytorch(
    args: Tuple[str, List[str]],
) -> Tuple[List[dict], float]:
    """Run GLiNER inference on a single chunk using PyTorch."""
    from gliner import GLiNER

    text, labels = args

    # Load model (will be cached after first load)
    model = GLiNER.from_pretrained("nvidia/gliner-PII")

    start = time.perf_counter()
    entities = model.predict_entities(text, labels)
    elapsed = (time.perf_counter() - start) * 1000

    return entities, elapsed


def run_single_inference_onnx(args: Tuple[str, List[str]]) -> Tuple[List[dict], float]:
    """Run GLiNER inference on a single chunk using ONNX."""
    from gliner import GLiNER

    text, labels = args

    # Load local ONNX model (INT8 quantized for CPU)
    model = GLiNER.from_pretrained(
        "models", load_onnx_model=True, onnx_model_file="model_quantized.onnx"
    )

    start = time.perf_counter()
    entities = model.predict_entities(text, labels)
    elapsed = (time.perf_counter() - start) * 1000

    return entities, elapsed


def benchmark_sequential(
    chunks: List[str], labels: List[str], use_onnx: bool = False
) -> dict:
    """Benchmark sequential processing of chunks."""
    from gliner import GLiNER

    # Load model once
    if use_onnx:
        model = GLiNER.from_pretrained(
            "models", load_onnx_model=True, onnx_model_file="model_quantized.onnx"
        )
    else:
        model = GLiNER.from_pretrained("nvidia/gliner-PII")

    # Warmup
    _ = model.predict_entities("Test warmup text.", labels)

    # Benchmark
    times = []
    all_entities = []

    for chunk in chunks:
        start = time.perf_counter()
        entities = model.predict_entities(chunk, labels)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        all_entities.extend(entities)

    return {
        "mode": "sequential",
        "backend": "onnx" if use_onnx else "pytorch",
        "num_chunks": len(chunks),
        "total_time_ms": sum(times),
        "avg_time_per_chunk_ms": sum(times) / len(times),
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "entities_found": len(all_entities),
    }


def benchmark_parallel_threads(
    chunks: List[str], labels: List[str], num_workers: int, use_onnx: bool = False
) -> dict:
    """Benchmark parallel processing using ThreadPoolExecutor."""
    from gliner import GLiNER

    # Pre-load model
    if use_onnx:
        model = GLiNER.from_pretrained(
            "models", load_onnx_model=True, onnx_model_file="model_quantized.onnx"
        )
    else:
        model = GLiNER.from_pretrained("nvidia/gliner-PII")

    # Warmup
    _ = model.predict_entities("Test warmup text.", labels)

    def process_chunk(chunk: str) -> Tuple[List[dict], float]:
        start = time.perf_counter()
        entities = model.predict_entities(chunk, labels)
        elapsed = (time.perf_counter() - start) * 1000
        return entities, elapsed

    # Benchmark
    start_total = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_chunk, chunks))

    total_time = (time.perf_counter() - start_total) * 1000

    times = [r[1] for r in results]
    all_entities = [e for r in results for e in r[0]]

    return {
        "mode": f"parallel_threads_{num_workers}",
        "backend": "onnx" if use_onnx else "pytorch",
        "num_workers": num_workers,
        "num_chunks": len(chunks),
        "total_time_ms": total_time,
        "avg_time_per_chunk_ms": sum(times) / len(times),
        "throughput_chunks_per_sec": len(chunks) / (total_time / 1000),
        "entities_found": len(all_entities),
    }


def benchmark_batch_inference(
    chunks: List[str], labels: List[str], batch_size: int, use_onnx: bool = False
) -> dict:
    """Benchmark batch inference (multiple chunks in one forward pass)."""
    from gliner import GLiNER

    # Load model
    if use_onnx:
        model = GLiNER.from_pretrained(
            "models", load_onnx_model=True, onnx_model_file="model_quantized.onnx"
        )
    else:
        model = GLiNER.from_pretrained("nvidia/gliner-PII")

    # Warmup
    _ = model.predict_entities("Test warmup text.", labels)

    # Process in batches using the batch inference method
    start_total = time.perf_counter()

    all_entities = []
    batch_times = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        start = time.perf_counter()
        # Use batch_predict_entities or predict_entities with list
        batch_results = model.batch_predict_entities(batch, labels)
        elapsed = (time.perf_counter() - start) * 1000

        batch_times.append(elapsed)
        for result in batch_results:
            all_entities.extend(result)

    total_time = (time.perf_counter() - start_total) * 1000

    return {
        "mode": f"batch_{batch_size}",
        "backend": "onnx" if use_onnx else "pytorch",
        "batch_size": batch_size,
        "num_chunks": len(chunks),
        "num_batches": len(batch_times),
        "total_time_ms": total_time,
        "avg_time_per_batch_ms": sum(batch_times) / len(batch_times),
        "avg_time_per_chunk_ms": total_time / len(chunks),
        "throughput_chunks_per_sec": len(chunks) / (total_time / 1000),
        "entities_found": len(all_entities),
    }


def generate_test_document(num_tokens: int = 10000) -> str:
    """Generate a synthetic test document with PII entities."""

    # Sample sentences with various PII types
    sentences = [
        "John Smith works at Microsoft Corporation in Seattle, Washington.",
        "Please contact jane.doe@example.com or call 555-123-4567 for more information.",
        "The patient John Williams was diagnosed with Type 2 Diabetes on 01/15/2024.",
        "Send payment to account number 1234567890 at First National Bank.",
        "Dr. Sarah Johnson, NPI 1234567890, prescribed Metformin 500mg.",
        "The defendant's SSN is 123-45-6789 and driver's license is D123-4567-8901.",
        "Our company address is 123 Main Street, New York, NY 10001.",
        "Meeting scheduled with Robert Chen from Acme Inc on March 15, 2024.",
        "Credit card ending in 4242 was charged $150.00 on 02/01/2024.",
        "Patient DOB: 05/20/1985, MRN: MRN-2024-001234.",
    ]

    # Repeat to reach target size
    result = []
    word_count = 0
    target_words = num_tokens  # Approximate tokens as words

    while word_count < target_words:
        for sentence in sentences:
            result.append(sentence)
            word_count += len(sentence.split())
            if word_count >= target_words:
                break

    return " ".join(result)


def main():
    parser = argparse.ArgumentParser(description="GLiNER CPU Parallel Benchmark")
    parser.add_argument(
        "--doc-size",
        type=int,
        default=10000,
        help="Document size in approximate tokens",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=450, help="Chunk size in tokens"
    )
    parser.add_argument(
        "--overlap", type=int, default=50, help="Overlap between chunks"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for batch inference"
    )
    parser.add_argument(
        "--use-onnx", action="store_true", help="Use ONNX backend instead of PyTorch"
    )
    args = parser.parse_args()

    # Labels to detect
    labels = [
        "person",
        "organization",
        "email",
        "phone",
        "ssn",
        "address",
        "date",
        "medical condition",
        "medication",
        "credit card",
    ]

    print("=" * 70)
    print("GLiNER CPU PARALLEL BENCHMARK")
    print("=" * 70)

    # System info
    info = get_system_info()
    print(f"\nSystem Info:")
    print(f"  CPU Cores: {info['cpu_count']}")
    print(f"  PyTorch Threads: {info['torch_threads']}")
    print(f"  Backend: {'ONNX' if args.use_onnx else 'PyTorch'}")

    # Generate test document
    print(f"\nGenerating {args.doc_size}-token test document...")
    document = generate_test_document(args.doc_size)

    # Chunk the document
    chunks = chunk_text(document, args.chunk_size, args.overlap)
    print(f"Document split into {len(chunks)} chunks")

    results = []

    # Test 1: Sequential processing
    print("\n[1/3] Running sequential benchmark...")
    seq_result = benchmark_sequential(chunks, labels, args.use_onnx)
    results.append(seq_result)
    print(
        f"  Total: {seq_result['total_time_ms']:.2f}ms, "
        f"Avg/chunk: {seq_result['avg_time_per_chunk_ms']:.2f}ms"
    )
    gc.collect()

    # Test 2: Parallel threads
    print(f"\n[2/3] Running parallel benchmark ({args.workers} workers)...")
    par_result = benchmark_parallel_threads(chunks, labels, args.workers, args.use_onnx)
    results.append(par_result)
    print(
        f"  Total: {par_result['total_time_ms']:.2f}ms, "
        f"Throughput: {par_result['throughput_chunks_per_sec']:.2f} chunks/sec"
    )
    gc.collect()

    # Test 3: Batch inference
    print(f"\n[3/3] Running batch benchmark (batch_size={args.batch_size})...")
    try:
        batch_result = benchmark_batch_inference(
            chunks, labels, args.batch_size, args.use_onnx
        )
        results.append(batch_result)
        print(
            f"  Total: {batch_result['total_time_ms']:.2f}ms, "
            f"Throughput: {batch_result['throughput_chunks_per_sec']:.2f} chunks/sec"
        )
    except Exception as e:
        print(f"  Batch inference failed: {e}")
        results.append({"mode": "batch", "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    # Speedup calculation
    if (
        len(results) >= 2
        and "total_time_ms" in results[0]
        and "total_time_ms" in results[1]
    ):
        speedup = results[0]["total_time_ms"] / results[1]["total_time_ms"]
        print(f"\nSpeedup from parallel: {speedup:.2f}x")

    return results


if __name__ == "__main__":
    main()
