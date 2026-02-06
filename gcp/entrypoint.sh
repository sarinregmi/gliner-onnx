#!/bin/bash
# ============================================================================
# Cloud Run Entrypoint - Syncs with Git on Startup
# ============================================================================
# This script pulls the latest benchmarking code from a public Git repo
# before starting the FastAPI server.
# ============================================================================

set -e

GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/sarinregmi/gliner-onnx.git}"
TEMP_DIR="/tmp/gliner_git_raw"

echo "============================================================"
echo "🚀 STARTUP: Syncing with Git"
echo "Target Repo: $GIT_REPO_URL"
echo "============================================================"
echo "🌍 ENVIRONMENT DIAGNOSTICS"
python3 -c "import numpy; print('Numpy version:', numpy.__version__)"
python3 -c "import torch; print('Torch version:', torch.__version__)"
echo "============================================================"

# Clone the repository to a temporary location
rm -rf "$TEMP_DIR"
git clone --depth 1 "$GIT_REPO_URL" "$TEMP_DIR"

# Copy ONLY the benchmark scripts into the working directory
# We DON'T copy the models/ or gliner/ folders because they are 
# already baked into the image and are too large to pull/overwrite.
echo "Updating benchmark scripts..."
cp "$TEMP_DIR"/*.py /app/
# Ensure we don't accidentally overwrite the local gliner folder if it's special
# but the scripts are what the user typically changes.

echo "✅ Sync complete."
echo "============================================================"

# Start the FastAPI server using uvicorn
# We use 'exec' so uvicorn becomes PID 1 and receives signals correctly
echo "Starting FastAPI server..."
exec uvicorn benchmark_wrapper_cpu:app --host 0.0.0.0 --port 8080
