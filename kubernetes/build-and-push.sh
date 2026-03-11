#!/usr/bin/env bash
# Build and push the torchtitan ROCm image to OCI Container Registry.
#
# Usage:
#   ./kubernetes/build-and-push.sh [TAG]
#
# Examples:
#   ./kubernetes/build-and-push.sh               # uses "rocm-latest"
#   ./kubernetes/build-and-push.sh rocm-20260311  # timestamped tag

set -euo pipefail

REGISTRY="iad.ocir.io/iduyx1qnmway"
IMAGE_NAME="torchtitan"
TAG="${1:-rocm-latest}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

# Build from repo root so COPY . . captures the full torchtitan source tree.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Building ${FULL_IMAGE} ..."
docker build \
  --file "${REPO_ROOT}/kubernetes/Dockerfile.rocm" \
  --tag "${FULL_IMAGE}" \
  "${REPO_ROOT}"

echo "Pushing ${FULL_IMAGE} ..."
docker push "${FULL_IMAGE}"

echo "Done: ${FULL_IMAGE}"
