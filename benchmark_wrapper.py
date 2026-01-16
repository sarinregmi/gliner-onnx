from fastapi import FastAPI, BackgroundTasks
import subprocess
import asyncio
import os

app = FastAPI()


@app.get("/")
def health_check():
    return {"status": "ready"}


@app.get("/run_benchmark")
async def run_benchmark(memory_load_gb: int = 0):
    """
    Triggers the benchmark script.
    Optional: memory_load_gb to simulate SLM usage.
    """

    # Run the benchmark as a subprocess and capture output
    command = ["python3", "benchmark.py"]
    if memory_load_gb > 0:
        command.extend(["--memory-load", str(memory_load_gb)])

    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        return {
            "status": "success" if process.returncode == 0 else "error",
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}
