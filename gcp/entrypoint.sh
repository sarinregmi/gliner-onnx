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

# Copy benchmark scripts AND the gliner library into the working directory
echo "Updating benchmark scripts and gliner library..."
cp "$TEMP_DIR"/*.py /app/
if [ -d "$TEMP_DIR/gliner" ]; then
    rm -rf /app/gliner
    cp -r "$TEMP_DIR/gliner" /app/
fi

echo "✅ Sync complete."
echo "============================================================"

# Start the FastAPI server using uvicorn
# We use 'exec' so uvicorn becomes PID 1 and receives signals correctly
echo "Starting FastAPI server..."
exec uvicorn benchmark_wrapper_cpu:app --host 0.0.0.0 --port 8080
