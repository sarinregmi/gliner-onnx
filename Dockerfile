# ============================================================================
# GLiNER Benchmark - Optimized Dockerfile for GCP Cloud Run with GPU
# ============================================================================
# Uses NVIDIA CUDA base image with Python for GPU-accelerated inference.
# Optimized for Cloud Run's NVIDIA L4 GPU (24GB VRAM).
# ============================================================================

# Use NVIDIA's official CUDA image with cuDNN for optimal GPU performance
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Hugging Face cache directory
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    # CUDA settings
    CUDA_VISIBLE_DEVICES=0 \
    # Cloud Run port
    PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA 12.1 support (matching the base image)
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    transformers==4.38.0 \
    huggingface_hub==0.20.3 \
    safetensors==0.4.2 \
    onnxruntime-gpu==1.17.0 \
    sentencepiece==0.2.0 \
    tqdm==4.66.1

# Install vLLM for SLM inference (Layer 3)
# Note: vLLM requires specific CUDA version and takes time to install
RUN pip install --no-cache-dir vllm==0.3.3

# Install GLiNER (from PyPI or local)
# Using local copy for custom modifications
COPY gliner /app/gliner
COPY gliner_config.json /app/

# Copy application code
COPY benchmark.py /app/
COPY benchmark_slm.py /app/
COPY benchmark_wrapper.py /app/
COPY requirements.txt /app/
COPY requirements-gpu.txt /app/

# Create cache directory for Hugging Face models
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# Pre-download the GLiNER model during build (optional but recommended)
# This adds ~1GB to the image but eliminates cold-start download time
RUN python -c "from gliner import GLiNER; GLiNER.from_pretrained('nvidia/gliner-PII')" || true

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
