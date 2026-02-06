#!/bin/bash
# ============================================================================
# GLiNER CPU-Only Benchmark - GCP Cloud Run Deployment Script
# ============================================================================
# Deploys the CPU-only version for cost-effective inference testing.
# No GPU required - much cheaper and faster to deploy.
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="gliner-benchmark-cpu"
IMAGE_NAME="gliner-benchmark-cpu"
MEMORY="4Gi"         # Less memory needed for CPU
CPU="8"              # Use 8 vCPUs for parallel processing
MIN_INSTANCES="0"    # Scale to zero
MAX_INSTANCES="2"    # Allow scaling for load testing
TIMEOUT="300"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    if ! command -v gcloud &> /dev/null; then
        echo "❌ gcloud CLI not found"
        exit 1
    fi
    echo "✅ gcloud CLI found"
    
    if [ "$PROJECT_ID" == "your-project-id" ]; then
        echo "❌ Please set GCP_PROJECT_ID"
        exit 1
    fi
    echo "✅ Project ID: $PROJECT_ID"
    
    gcloud config set project "$PROJECT_ID"
}

enable_apis() {
    print_header "Enabling Required APIs"
    
    gcloud services enable \
        run.googleapis.com \
        artifactregistry.googleapis.com \
        cloudbuild.googleapis.com \
        --quiet
    
    echo "✅ APIs enabled"
}

create_artifact_registry() {
    print_header "Creating Artifact Registry Repository"
    
    if gcloud artifacts repositories describe docker-repo --location="$REGION" &> /dev/null; then
        echo "✅ Repository already exists"
    else
        gcloud artifacts repositories create docker-repo \
            --repository-format=docker \
            --location="$REGION" \
            --description="Docker repository for GLiNER benchmark"
        echo "✅ Repository created"
    fi
}

build_and_push_image() {
    print_header "Building CPU-Only Docker Image"
    
    IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/docker-repo/${IMAGE_NAME}:latest"
    
    echo "Building image: $IMAGE_URI"
    echo "Using Dockerfile.cpu (no CUDA dependencies)..."
    
    # Build using the CPU-only Dockerfile
    gcloud builds submit \
        --tag "$IMAGE_URI" \
        --timeout=1200s \
        --machine-type=e2-highcpu-8 \
        -f Dockerfile.cpu \
        ..
    
    echo "✅ Image built and pushed: $IMAGE_URI"
}

deploy_to_cloud_run() {
    print_header "Deploying to Cloud Run (CPU-Only)"
    
    IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/docker-repo/${IMAGE_NAME}:latest"
    
    # Note: No --gpu flag for CPU-only deployment
    gcloud run deploy "$SERVICE_NAME" \
        --image "$IMAGE_URI" \
        --region "$REGION" \
        --platform managed \
        --memory "$MEMORY" \
        --cpu "$CPU" \
        --min-instances "$MIN_INSTANCES" \
        --max-instances "$MAX_INSTANCES" \
        --timeout "$TIMEOUT" \
        --allow-unauthenticated \
        --set-env-vars="PYTHONUNBUFFERED=1,OMP_NUM_THREADS=2,MKL_NUM_THREADS=2" \
        --port 8080
    
    echo "✅ Deployed to Cloud Run (CPU-Only)"
}

get_service_url() {
    print_header "Getting Service URL"
    
    SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --region "$REGION" \
        --format="value(status.url)")
    
    echo ""
    echo "🎉 CPU-Only Deployment Complete!"
    echo ""
    echo "Service URL: $SERVICE_URL"
    echo ""
    echo "Test endpoints:"
    echo "  Health:     curl \"$SERVICE_URL/\""
    echo "  CPU Info:   curl \"$SERVICE_URL/cpu_info\""
    echo "  Quick Test: curl \"$SERVICE_URL/quick_test\""
    echo "  Benchmark:  curl \"$SERVICE_URL/run_benchmark?doc_size=10000&workers=4&use_onnx=true\""
    echo ""
}

# ============================================================================
# MAIN
# ============================================================================
main() {
    print_header "GLiNER CPU-Only Benchmark - GCP Cloud Run Deployment"
    
    check_prerequisites
    enable_apis
    create_artifact_registry
    build_and_push_image
    deploy_to_cloud_run
    get_service_url
}

main "$@"
