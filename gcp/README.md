# GCP Cloud Run Deployment Guide

This directory contains everything needed to deploy the GLiNER benchmark to **Google Cloud Run with GPU** (NVIDIA L4).

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **gcloud CLI** installed and authenticated:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

## Quick Start

### Option 1: Automated Deployment (Recommended)

```bash
# Set your project ID
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"

# Run the deployment script
chmod +x deploy.sh
./deploy.sh
```

### Option 2: Manual Deployment

```bash
# 1. Enable required APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# 2. Create Artifact Registry repository
gcloud artifacts repositories create docker-repo \
    --repository-format=docker \
    --location=us-central1

# 3. Build and push image using Cloud Build
gcloud builds submit \
    --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/docker-repo/gliner-benchmark:latest \
    --timeout=1800s \
    ..

# 4. Deploy to Cloud Run with GPU
gcloud run deploy gliner-benchmark \
    --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/docker-repo/gliner-benchmark:latest \
    --region us-central1 \
    --platform managed \
    --memory 16Gi \
    --cpu 4 \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --min-instances 0 \
    --max-instances 1 \
    --timeout 300 \
    --allow-unauthenticated \
    --port 8080
```

## Running Benchmarks

Once deployed, you'll get a service URL like: `https://gliner-benchmark-xxxxx-uc.a.run.app`

### Basic Benchmark
```bash
curl "https://YOUR_SERVICE_URL/run_benchmark"
```

### With Memory Simulation (Simulates SLM Load)
```bash
# Simulate 10GB of GPU memory used by an SLM
curl "https://YOUR_SERVICE_URL/run_benchmark?memory_load_gb=10"
```

### Health Check
```bash
curl "https://YOUR_SERVICE_URL/"
```

## Resource Specifications

| Resource | Value | Notes |
|----------|-------|-------|
| GPU | NVIDIA L4 | 24GB VRAM, Tensor Cores |
| Memory | 16 GiB | Container RAM |
| CPU | 4 vCPUs | Dedicated cores |
| Timeout | 300s | Max request duration |
| Min Instances | 0 | Scale to zero when idle |
| Max Instances | 1 | For benchmarking |

## Cost Estimation

- **GPU (L4)**: ~$0.70/hour while running
- **Scale to Zero**: $0 when no requests
- **Cold Start**: ~30-60 seconds (includes model loading)

## Cleanup

```bash
# Delete the Cloud Run service
gcloud run services delete gliner-benchmark --region us-central1

# Delete the Artifact Registry repository (optional)
gcloud artifacts repositories delete docker-repo --location us-central1
```

## Troubleshooting

### Build Timeout
If the build times out, increase the timeout:
```bash
gcloud builds submit --timeout=3600s ...
```

### GPU Not Available
Ensure GPU is enabled in your region. L4 GPUs are available in:
- us-central1
- us-east1
- us-west1
- europe-west4

### Memory Issues
If you see OOM errors, the models don't fit in the allocated memory. Increase `--memory` to `32Gi`.
