"""
GLiNER Benchmark Wrapper for Cloud Run
Exposes the benchmark script as HTTP endpoints for triggering via curl.
"""

from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import subprocess
import asyncio
import os
import torch
import time
from typing import List

# Global model variable for caching
GLOBAL_MODEL = None

app = FastAPI(
    title="GLiNER Benchmark",
    description="GPU-accelerated GLiNER NER benchmark on Cloud Run",
    version="1.0.0",
)

def load_model():
    """Load model into global variable if not already loaded."""
    global GLOBAL_MODEL
    if GLOBAL_MODEL is None:
        try:
            from gliner import GLiNER
            model_path = "/app/models"
            if os.path.exists(os.path.join(model_path, "model.onnx")):
                print(f"Loading ONNX model from {model_path}...")
                GLOBAL_MODEL = GLiNER.from_pretrained(model_path, load_onnx_model=True)
                print("Model loaded successfully.")
            else:
                print("ONNX model not found, cannot load.")
        except Exception as e:
            print(f"Failed to load model: {e}")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

class PredictRequest(BaseModel):
    text: str
    labels: List[str]

@app.post("/predict")
async def predict(request: PredictRequest = Body(...)):
    """
    Run inference on the loaded model.
    """
    global GLOBAL_MODEL
    if GLOBAL_MODEL is None:
        load_model()
    
    if GLOBAL_MODEL is None:
        return JSONResponse(
            {"status": "error", "error": "Model failed to load"}, 
            status_code=500
        )

    try:
        start_time = time.perf_counter()
        entities = GLOBAL_MODEL.predict_entities(request.text, request.labels)
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "entities": entities,
            "inference_time_ms": inference_time_ms
        }
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)



@app.get("/")
def health_check():
    """Health check endpoint with GPU info."""
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
        gpu_info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )

    return {"status": "ready", "gpu": gpu_info}


@app.get("/gpu_info")
def gpu_info():
    """Detailed GPU information."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "multi_processor_count": props.multi_processor_count,
        "major": props.major,
        "minor": props.minor,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


@app.get("/run_benchmark")
async def run_benchmark(
    memory_load_gb: int = Query(
        default=0,
        ge=0,
        le=20,
        description="GB of GPU memory to pre-allocate (simulates SLM usage)",
    )
):
    """
    Triggers the benchmark script.

    Args:
        memory_load_gb: Optional GB of GPU memory to occupy before testing.
                        Use this to simulate running alongside an SLM model.
    """
    command = ["python", "benchmark.py"]
    if memory_load_gb > 0:
        command.extend(["--memory-load", str(memory_load_gb)])

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/app",
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=280  # Cloud Run default timeout is 300s
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
        return JSONResponse({"status": "failed", "error": str(e)}, status_code=500)


@app.get("/warmup")
async def warmup():
    """Pre-load the GLiNER model to reduce cold start latency."""
    try:
        global GLOBAL_MODEL
        if GLOBAL_MODEL is None:
            load_model()
            
        if GLOBAL_MODEL is None:
             return JSONResponse(
                {"status": "error", "error": "Model failed to load"}, 
                status_code=500
            )
        
        # Run a quick inference
        result = GLOBAL_MODEL.predict_entities(
            "John Doe works at Microsoft.", ["person", "organization"]
        )

        return {
            "status": "warmed_up",
            "device": "onnx",
            "test_entities_found": len(result),
        }
    except Exception as e:
        return JSONResponse(
            {"status": "warmup_failed", "error": str(e)}, status_code=500
        )


@app.get("/run_slm_benchmark")
async def run_slm_benchmark(
    model: str = Query(
        default="microsoft/Phi-3-mini-4k-instruct",
        description="Model name to benchmark",
    ),
    runs: int = Query(default=10, ge=1, le=50, description="Number of benchmark runs"),
):
    """
    Run SLM (Layer 3) benchmark using vLLM.
    Tests inference latency for semantic analysis models like Phi-3/4-mini.
    """
    command = [
        "python",
        "benchmark_slm.py",
        "--model",
        model,
        "--runs",
        str(runs),
        "--backend",
        "vllm",
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/app",
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=280)

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
            {"status": "timeout", "error": "SLM benchmark exceeded 280 second limit"},
            status_code=504,
        )
    except Exception as e:
        return JSONResponse({"status": "failed", "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
