#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="productdaydemo/opencv-vision:latest"
ARCHIVE="$(mktemp --suffix=.tar)"
CA_FILE="$SCRIPT_DIR/foundry-ca.crt"
trap 'unset FOUNDRY_API_KEY; sudo rm -f "$ARCHIVE" "$CA_FILE"' EXIT

FOUNDRY_API_KEY="$(
  sudo kubectl -n foundry-local-operator get secret candy-yolo11s-api-keys \
    -o jsonpath='{.data.primary-key}' | base64 -d
)"
test -n "$FOUNDRY_API_KEY"
sudo kubectl -n productdaydemo create secret generic candy-yolo11s-client \
  --from-literal=api-key="$FOUNDRY_API_KEY" \
  --dry-run=client -o yaml | sudo kubectl apply -f -
unset FOUNDRY_API_KEY

sudo kubectl -n foundry-local-operator get configmap foundry-local-operator-ca-bundle \
  -o jsonpath='{.data.ca\.crt}' > "$CA_FILE"
test -s "$CA_FILE"
sudo kubectl -n productdaydemo create configmap foundry-local-ca \
  --from-file=ca.crt="$CA_FILE" \
  --dry-run=client -o yaml | sudo kubectl apply -f -
rm -f "$CA_FILE"

sudo docker build --network=host -f "$SCRIPT_DIR/Dockerfile" -t "$IMAGE" "$PROJECT_DIR"
sudo docker run --rm "$IMAGE" python -m unittest discover -s /app/tests -v
sudo docker save "$IMAGE" -o "$ARCHIVE"
sudo k3s ctr images import "$ARCHIVE"
sudo kubectl apply -f "$SCRIPT_DIR/k8s.yaml"
sudo kubectl -n productdaydemo rollout restart deployment/opencv-vision
sudo kubectl -n productdaydemo rollout status deployment/opencv-vision --timeout=10m

echo "OpenCV UI: http://$(hostname -I | awk '{print $1}'):8094/"
