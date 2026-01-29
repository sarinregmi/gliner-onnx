# ============================================================================
# GLiNER Benchmark - Optimized Dockerfile for GCP Cloud Run (CPU Only)
# ============================================================================
# Uses Python 3.10 slim image.
# Optimized for Cloud Run with CPU (ONNX Runtime).
# ============================================================================

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Hugging Face cache directory
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    # Cloud Run port
    PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
# numpy<2.0 is required to prevent "A module that was compiled using NumPy 1.x" errors with PyTorch
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    transformers==4.40.0 \
    huggingface_hub==0.20.3 \
    safetensors==0.4.2 \
    onnx==1.15.0 \
    onnxruntime==1.17.0 \
    sentencepiece==0.2.0 \
    tqdm==4.66.1

# Install GLiNER (from PyPI or local)
# Using local copy for custom modifications
COPY gliner /app/gliner
COPY gliner_config.json /app/

# Copy application code
COPY benchmark.py /app/
COPY benchmark_slm.py /app/
COPY benchmark_wrapper.py /app/
COPY convert_model.py /app/
COPY verify_model.py /app/
COPY requirements.txt /app/

# Create cache directory for Hugging Face models
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# Convert the model to ONNX during build
# This ensures model.onnx is available in /app/models
RUN python convert_model.py

# Verify the converted model works
RUN python verify_model.py

# Create a non-root user (Cloud Run best practice)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Expose the port Cloud Run expects
EXPOSE 8080

# Run the FastAPI application
CMD ["uvicorn", "benchmark_wrapper:app", "--host", "0.0.0.0", "--port", "8080"]
