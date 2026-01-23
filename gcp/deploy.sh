#!/bin/bash
# ============================================================================
# GLiNER Benchmark - GCP Cloud Run Deployment Script
# ============================================================================
# This script deploys the GLiNER benchmark environment to Google Cloud Run
# with GPU support (NVIDIA L4).
#
# Prerequisites:
# 1. gcloud CLI installed and authenticated
# 2. Docker installed (for local builds) OR use Cloud Build
# 3. Artifact Registry API enabled
# 4. Cloud Run API enabled
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION - Modify these values for your environment
# ============================================================================
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="gliner-benchmark"
IMAGE_NAME="gliner-benchmark"
MEMORY="16Gi"        # Memory allocation
CPU="4"              # vCPUs
GPU_TYPE="nvidia-l4" # GPU type (L4 is standard for Cloud Run)
GPU_COUNT="1"        # Number of GPUs
MIN_INSTANCES="0"    # Scale to zero when idle
MAX_INSTANCES="1"    # Max instances for benchmark
TIMEOUT="300"        # Request timeout in seconds

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
    
    # Check gcloud
    if ! command -v gcloud &> /dev/null; then
        echo "❌ gcloud CLI not found. Please install: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    echo "✅ gcloud CLI found"
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 | grep -q "@"; then
        echo "❌ Not authenticated. Run: gcloud auth login"
        exit 1
    fi
    echo "✅ Authenticated to GCP"
    
    # Check project
    if [ "$PROJECT_ID" == "your-project-id" ]; then
        echo "❌ Please set GCP_PROJECT_ID environment variable or edit this script"
        exit 1
    fi
    echo "✅ Project ID: $PROJECT_ID"
    
    # Set project
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
    
    # Check if repository exists
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
    print_header "Building and Pushing Docker Image"
    
    IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/docker-repo/${IMAGE_NAME}:latest"
    
    echo "Building image: $IMAGE_URI"
    echo "This may take 10-15 minutes for the first build..."
    
    # Use Cloud Build for remote building (no local Docker needed)
    gcloud builds submit \
        --tag "$IMAGE_URI" \
        --timeout=1800s \
        --machine-type=e2-highcpu-8 \
        ..
    
    echo "✅ Image built and pushed: $IMAGE_URI"
}

deploy_to_cloud_run() {
    print_header "Deploying to Cloud Run"
    
    IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/docker-repo/${IMAGE_NAME}:latest"
    
    gcloud run deploy "$SERVICE_NAME" \
        --image "$IMAGE_URI" \
        --region "$REGION" \
        --platform managed \
        --memory "$MEMORY" \
        --cpu "$CPU" \
        --gpu "$GPU_COUNT" \
        --gpu-type "$GPU_TYPE" \
        --min-instances "$MIN_INSTANCES" \
        --max-instances "$MAX_INSTANCES" \
        --timeout "$TIMEOUT" \
        --allow-unauthenticated \
        --set-env-vars="PYTHONUNBUFFERED=1" \
        --port 8080
    
    echo "✅ Deployed to Cloud Run"
}

get_service_url() {
    print_header "Getting Service URL"
    
    SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --region "$REGION" \
        --format="value(status.url)")
    
    echo ""
    echo "🎉 Deployment Complete!"
    echo ""
    echo "Service URL: $SERVICE_URL"
    echo ""
    echo "To run the benchmark:"
    echo "  curl \"$SERVICE_URL/run_benchmark\""
    echo ""
    echo "To run with memory simulation (simulates SLM load):"
    echo "  curl \"$SERVICE_URL/run_benchmark?memory_load_gb=10\""
    echo ""
}

cleanup() {
    print_header "Cleanup (Optional)"
    
    echo "To delete the Cloud Run service:"
    echo "  gcloud run services delete $SERVICE_NAME --region $REGION"
    echo ""
    echo "To delete the Artifact Registry repository:"
    echo "  gcloud artifacts repositories delete docker-repo --location $REGION"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================
main() {
    print_header "GLiNER Benchmark - GCP Cloud Run Deployment"
    
    check_prerequisites
    enable_apis
    create_artifact_registry
    build_and_push_image
    deploy_to_cloud_run
    get_service_url
    cleanup
}

# Run main function
main "$@"
