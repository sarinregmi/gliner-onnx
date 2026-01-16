
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and basic tools
RUN apt-get update && apt-get install -y python3 python3-pip git wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch and Dependencies
# Note: Installing PyTorch with CUDA 11.8 support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir fastapi uvicorn gliner transformers onnxruntime-gpu vllm

# Copy project files
COPY . /app

# Expose port (Cloud Run defaults to 8080)
ENV PORT=8080
EXPOSE 8080

# Run the wrapper app
CMD ["uvicorn", "benchmark_wrapper:app", "--host", "0.0.0.0", "--port", "8080"]
