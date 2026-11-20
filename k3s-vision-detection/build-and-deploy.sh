#!/bin/bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "YOLO deployment disabled for the OpenCV demo; no cluster changes made."
  echo "To explicitly enable YOLO, rerun with --enable-yolo."
  exit 0
fi
if [ "$#" -ne 1 ] || [ "$1" != "--enable-yolo" ]; then
  echo "Usage: $0 [--enable-yolo]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="productdaydemo/vision-detection:latest"
ARCHIVE="$(mktemp --suffix=.tar)"
CA_FILE="$(mktemp --suffix=.crt)"
trap 'unset FOUNDRY_API_KEY; sudo rm -f "$ARCHIVE" "$CA_FILE"' EXIT

sudo kubectl create namespace productdaydemo --dry-run=client -o yaml | sudo kubectl apply -f -
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

sudo docker build --network=host -t "$IMAGE" "$SCRIPT_DIR"
sudo docker run --rm "$IMAGE" python -m unittest discover -s /app/tests -v
sudo docker save "$IMAGE" -o "$ARCHIVE"
sudo k3s ctr images import "$ARCHIVE"
sudo kubectl apply -f "$SCRIPT_DIR/k8s.yaml"
sudo kubectl -n productdaydemo scale deployment/vision-detection --replicas=1
sudo kubectl -n productdaydemo rollout restart deployment/vision-detection
sudo kubectl -n productdaydemo rollout status deployment/vision-detection --timeout=10m
