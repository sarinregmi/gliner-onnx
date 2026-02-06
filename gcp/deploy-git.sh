#!/bin/bash
# ============================================================================
# GLiNER Git-Sync Deployment Script (GCP Cloud Run)
# ============================================================================
# Allows fast code updates without rebuilding the heavy model image.
# ============================================================================

set -e

PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="gliner-cpu-final"
IMAGE_NAME="gliner-benchmark-git"

# Default configuration
MEMORY="2Gi"
CPU="1"
MAX_INSTANCES="2"

print_header() {
    echo "------------------------------------------------------------"
    echo "$1"
    echo "------------------------------------------------------------"
}

usage() {
    echo "Usage: ./deploy-git.sh [command]"
    echo ""
    echo "Commands:"
    echo "  full-build   Build the Docker image (including model) and deploy."
    echo "               Do this the first time or when dependencies change."
    echo "  update-code  Trigger Cloud Run to restart and pull latest code from Git."
    echo "               Instant deployment (no build needed)."
    echo ""
}

check_project() {
    if [ "$PROJECT_ID" == "your-project-id" ]; then
        echo "❌ Error: GCP_PROJECT_ID is not set."
        exit 1
    fi
    echo "✅ Setting project to $PROJECT_ID"
    gcloud config set project "$PROJECT_ID"
}

enable_apis() {
    print_header "Enabling Required APIs"
    gcloud services enable \
        run.googleapis.com \
        artifactregistry.googleapis.com \
        cloudbuild.googleapis.com \
        --quiet
}

create_artifact_registry() {
    print_header "Checking Artifact Registry"
    if gcloud artifacts repositories describe docker-repo --location="$REGION" &> /dev/null; then
        echo "✅ Repository already exists"
    else
        gcloud artifacts repositories create docker-repo \
            --repository-format=docker \
            --location="$REGION" \
            --description="Docker repository for GLiNER benchmark"
    fi
}

deploy() {
    print_header "Deploying to Cloud Run"
    
    # Use IMAGE_URI if set, otherwise default to latest
    if [ -z "$IMAGE_URI" ]; then
        IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/docker-repo/${IMAGE_NAME}:latest"
    fi
    
    gcloud run deploy "$SERVICE_NAME" \
        --image "$IMAGE_URI" \
        --region "$REGION" \
        --platform managed \
        --memory "$MEMORY" \
        --cpu "$CPU" \
        --max-instances "$MAX_INSTANCES" \
        --allow-unauthenticated \
        --set-env-vars="GIT_REPO_URL=https://github.com/sarinregmi/gliner-onnx.git" \
        --port 8080
}

case "$1" in
    "full-build")
        check_project
        enable_apis
        create_artifact_registry
        print_header "Building Heavy Image (including models/)"
        
        TAG=$(date +%Y%m%d%H%M%S)
        IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/docker-repo/${IMAGE_NAME}:${TAG}"
        LATEST_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/docker-repo/${IMAGE_NAME}:latest"
        
        # Build using the Git-enabled Dockerfile
        # Note: we temporarily rename cloudbuild.yaml so gcloud uses the Dockerfile
        [ -f cloudbuild.yaml ] && mv cloudbuild.yaml cloudbuild.yaml.bak
        cp Dockerfile.git Dockerfile
        
        gcloud builds submit \
            --tag "$IMAGE_URI" \
            --timeout=1200s \
            .
            
        # Also tag as latest
        gcloud artifacts docker tags add "$IMAGE_URI" "$LATEST_URI" --quiet
        
        rm Dockerfile
        [ -f cloudbuild.yaml.bak ] && mv cloudbuild.yaml.bak cloudbuild.yaml
        
        # Deploy using the SPECIFIC tag to force update
        IMAGE_URI="$IMAGE_URI" deploy
        ;;
        
    "update-code")
        check_project
        enable_apis
        print_header "Refreshing Code from Git (Fast Restart)"
        
        # We trigger a new revision using the SAME image.
        # This causes Cloud Run to start new containers, which run entrypoint.sh.
        # entrypoint.sh pulls the latest code from GitHub.
        
        gcloud run services update "$SERVICE_NAME" \
            --region "$REGION" \
            --update-env-vars="LAST_UPDATE=$(date +%s)"
            
        echo "✅ Service updated. New instances will pull the latest Git code."
        ;;
        
    *)
        usage
        exit 1
        ;;
esac
