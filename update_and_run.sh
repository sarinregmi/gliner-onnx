#!/bin/bash
echo "Updating code from GitHub..."
cd /home/ubuntu/gliner-onnx
git pull origin main
source venv/bin/activate

# Reinstall requirements based on hardware
if lspci | grep -i nvidia; then
   echo "GPU detected, using GPU requirements"
   pip uninstall -y onnxruntime onnxruntime-gpu
   pip install -r requirements-gpu.txt
else
   pip install -r requirements.txt
fi

echo "Running benchmark..."
python3 benchmark.py
