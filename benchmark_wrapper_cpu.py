"""
GLiNER CPU Benchmark Wrapper for Cloud Run
Exposes CPU benchmark endpoints via HTTP.
"""

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import asyncio
import os
import multiprocessing as mp

app = FastAPI(
    title="GLiNER CPU Benchmark",
    description="CPU-only GLiNER benchmark on Cloud Run",
    version="1.0.0",
)


@app.get("/")
def health_check():
    """Health check endpoint with CPU info."""
    import torch

    return {
        "status": "ready",
        "cpu": {
            "core_count": mp.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
        "gpu": {"available": False, "reason": "CPU-only deployment"},
    }


@app.get("/cpu_info")
def cpu_info():
    """Detailed CPU information."""
    import torch
    import platform
    import numpy as np

    return {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": mp.cpu_count(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "omp_threads": os.environ.get("OMP_NUM_THREADS", "default"),
        "mkl_threads": os.environ.get("MKL_NUM_THREADS", "default"),
    }


@app.get("/run_benchmark")
async def run_benchmark(
    doc_size: int = Query(
        default=10000, ge=100, le=100000, description="Document size in tokens"
    ),
    workers: int = Query(
        default=4, ge=1, le=16, description="Number of parallel workers"
    ),
    batch_size: int = Query(
        default=8, ge=1, le=32, description="Batch size for batch inference"
    ),
    use_onnx: bool = Query(
        default=True, description="Use ONNX backend (faster on CPU)"
    ),
):
    """
    Run the CPU parallel benchmark.
    Tests sequential, parallel, and batch processing strategies.
    """
    command = [
        "python",
        "benchmark_cpu_parallel.py",
        "--doc-size",
        str(doc_size),
        "--workers",
        str(workers),
        "--batch-size",
        str(batch_size),
    ]

    if use_onnx:
        command.append("--use-onnx")

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/app",
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=280,
        )

        return JSONResponse(
            {
                "status": "success" if process.returncode == 0 else "error",
                "return_code": process.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode() if stderr else None,
            }
        )

    except asyncio.TimeoutError:
        return JSONResponse(
            {"status": "timeout", "error": "Benchmark exceeded 280 second limit"},
            status_code=504,
        )
    except Exception as e:
        return JSONResponse(
            {"status": "failed", "error": str(e)},
            status_code=500,
        )


@app.get("/warmup")
async def warmup():
    """Pre-load the GLiNER model."""
    try:
        from gliner import GLiNER

        # Load local ONNX model (INT8 quantized for CPU)
        model = GLiNER.from_pretrained(
            "models", load_onnx_model=True, onnx_model_file="model_quantized.onnx"
        )

        # Run inference
        result = model.predict_entities(
            "John Doe works at Microsoft.", ["person", "organization"]
        )

        return {
            "status": "warmed_up",
            "backend": "onnx",
            "device": "cpu",
            "test_entities_found": len(result),
        }
    except Exception as e:
        return JSONResponse(
            {"status": "warmup_failed", "error": str(e)},
            status_code=500,
        )


@app.get("/quick_test")
async def quick_test(
    text: str = Query(
        default="John Doe works at Microsoft Corp. Email: john@example.com",
        description="Text to analyze",
    ),
    use_onnx: bool = Query(default=True, description="Use ONNX backend"),
):
    """Quick inference test on provided text."""
    import time

    try:
        from gliner import GLiNER

        labels = ["person", "organization", "email", "phone", "address"]

        # Load model
        load_start = time.perf_counter()
        if use_onnx:
            model = GLiNER.from_pretrained(
                "models", load_onnx_model=True, onnx_model_file="model_quantized.onnx"
            )
        else:
            model = GLiNER.from_pretrained("nvidia/gliner-PII")
        load_time = (time.perf_counter() - load_start) * 1000

        # Inference
        infer_start = time.perf_counter()
        entities = model.predict_entities(text, labels)
        infer_time = (time.perf_counter() - infer_start) * 1000

        return {
            "status": "success",
            "backend": "onnx" if use_onnx else "pytorch",
            "load_time_ms": round(load_time, 2),
            "inference_time_ms": round(infer_time, 2),
            "entities": entities,
        }
    except Exception as e:
        return JSONResponse(
            {"status": "failed", "error": str(e)},
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
