
# GLiNER ONNX Project

This project contains:
1.  A patched, local version of the GLiNER library (supporting `export_to_onnx`).
2.  Scripts to convert the GLiNER-PII model to ONNX.
3.  Scripts to verify and use the converted ONNX model.
4.  Pre-converted ONNX models (standard and quantized).

## Directory Structure
-   `venv/`: Virtual environment (activate with `source venv/bin/activate`)
-   `gliner/`: Local patched library source code.
-   `models/`: Directory containing the converted ONNX models.
-   `convert_model.py`: Script to convert the HuggingFace model to ONNX.
-   `verify_model.py`: Script to load the ONNX model and run a test prediction.
-   `benchmark.py`: Script to compare performance across all model variants.

## Usage

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. Install Dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

### 3. Run Benchmark
```bash
python benchmark.py
```

### 4. Verify the Model
Run the verification script to load the pre-converted model in `models/` and check if it predicts entities correctly.
```bash
python verify_model.py
```

### 3. Convert a Model (Optional)
If you need to re-convert the model:
```bash
python convert_model.py
```
This will overwrite the files in `models/`.

## Why Local GLiNER?
The version of `gliner` on PyPI (0.2.24) was found to be missing the `export_to_onnx` functionality, even though correct version numbers were reported. This project bundles a local version of the library (also versioned 0.2.24) that includes the necessary code for ONNX export.
