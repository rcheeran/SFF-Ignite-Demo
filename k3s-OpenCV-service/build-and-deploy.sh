#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="productdaydemo/opencv-vision:latest"
ARCHIVE="$(mktemp --suffix=.tar)"
trap 'sudo rm -f "$ARCHIVE"' EXIT

sudo docker build --network=host -f "$SCRIPT_DIR/Dockerfile" -t "$IMAGE" "$PROJECT_DIR"
sudo docker run --rm "$IMAGE" python -m unittest discover -s /app/tests -v
sudo docker save "$IMAGE" -o "$ARCHIVE"
sudo k3s ctr images import "$ARCHIVE"
sudo kubectl apply -f "$SCRIPT_DIR/k8s.yaml"
sudo kubectl -n productdaydemo rollout restart deployment/opencv-vision
sudo kubectl -n productdaydemo rollout status deployment/opencv-vision --timeout=10m

echo "OpenCV UI: http://$(hostname -I | awk '{print $1}'):8094/"
