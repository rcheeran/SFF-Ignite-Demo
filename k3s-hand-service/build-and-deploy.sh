#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="productdaydemo/hand-landmarker:latest"
ARCHIVE="$(mktemp --suffix=.tar)"
trap 'sudo rm -f "$ARCHIVE"' EXIT

sudo docker build --network=host -t "$IMAGE" "$SCRIPT_DIR"
sudo docker run --rm \
    -v "$SCRIPT_DIR:/tests:ro" \
    -w /tests \
    "$IMAGE" \
    python -m unittest -v test_hand_logic.py test_app.py
sudo docker save "$IMAGE" -o "$ARCHIVE"
sudo k3s ctr images import "$ARCHIVE"
sudo kubectl apply -f "$SCRIPT_DIR/k8s.yaml"
sudo kubectl -n productdaydemo rollout restart deployment/hand-landmarker
sudo kubectl -n productdaydemo rollout status deployment/hand-landmarker --timeout=10m

echo "Hand Landmarker UI: http://$(hostname -I | awk '{print $1}'):8095/"
