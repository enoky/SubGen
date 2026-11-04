#!/usr/bin/env bash
# Build SubGen with Cloud Build and deploy it to Cloud Run on one NVIDIA L4 GPU.
# Run from the root of the SubGen repo:  bash cloudrun/deploy.sh
# Override any setting inline, e.g.  REGION=us-central1 bash cloudrun/deploy.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-east4}"          # L4 regions: us-east4, us-central1, europe-west1, europe-west4, asia-southeast1
REPO="${REPO:-subgen}"                # Artifact Registry repository
SERVICE="${SERVICE:-subgen}"
WHISPER_MODELS="${WHISPER_MODELS:-large-v3}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/subgen:latest"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Set a project first: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud artifacts repositories describe "${REPO}" --location "${REGION}" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "${REPO}" --repository-format docker --location "${REGION}"

# The image is large (CUDA wheels plus ~6 GB of models), so use a bigger build machine and disk.
cat > /tmp/subgen-cloudbuild.yaml <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args: ["build", "--build-arg", "WHISPER_MODELS=${WHISPER_MODELS}", "-t", "${IMAGE}", "."]
images: ["${IMAGE}"]
timeout: 3600s
options:
  machineType: E2_HIGHCPU_8
  diskSizeGb: 200
EOF
gcloud builds submit --config /tmp/subgen-cloudbuild.yaml .

gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy \
  --cpu 8 --memory 32Gi --no-cpu-throttling \
  --concurrency 4 --min-instances 0 --max-instances 1 \
  --timeout 3600 --session-affinity \
  --no-allow-unauthenticated

echo
echo "Deployed privately. Open it through an authenticated tunnel with:"
echo "  gcloud run services proxy ${SERVICE} --region ${REGION} --port 8080"
echo "then browse to http://localhost:8080"
