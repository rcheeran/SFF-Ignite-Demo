#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="productdaydemo/owlvit-vision:latest"
ARCHIVE="$(mktemp --suffix=.tar)"
trap 'sudo rm -f "$ARCHIVE"' EXIT

sudo docker build --network=host -t "$IMAGE" "$SCRIPT_DIR"
sudo docker save "$IMAGE" -o "$ARCHIVE"
sudo k3s ctr images import "$ARCHIVE"
sudo kubectl apply -f "$SCRIPT_DIR/k8s.yaml"
sudo kubectl -n productdaydemo rollout restart deployment/owlvit-vision
sudo kubectl -n productdaydemo rollout status deployment/owlvit-vision --timeout=10m

echo "Vision UI: http://$(hostname -I | awk '{print $1}'):8094/"
