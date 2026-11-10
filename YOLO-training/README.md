# Candy YOLO11 training

This directory fine-tunes `yolo11s.pt` on the 11-class candy dataset. Large
artifacts live on the k3s data mount through the `storage` symlink.
`k3s-OpenCV-service/datasets` points to the same mount. Preparation creates
hard-linked train and validation splits to avoid duplicating image data.

The source dataset spells class 8 as `starbust`. The generated `data.yaml`
corrects its display name to `starburst` without changing its numeric class ID.

## Prepare

```bash
cd ~/SFF-Ignite-Demo/YOLO-training
python prepare_dataset.py
```

## Train on the Jetson Thor GPU

The local `clouduser` account cannot read `/dev/nvmap`, so CUDA training must
run through the privileged Python process:

```bash
cd ~/SFF-Ignite-Demo/YOLO-training/storage
sudo env \
  YOLO_CONFIG_DIR="$PWD/config" \
  TORCH_HOME="$PWD/cache/torch" \
  XDG_CACHE_HOME="$PWD/cache" \
  /home/clouduser/venv/bin/python \
  /home/clouduser/SFF-Ignite-Demo/YOLO-training/train.py
```

Defaults follow the referenced Ultralytics workflow:

- Pretrained YOLO11s weights
- 640 x 640 input
- 80 epochs with early stopping after 20 epochs without improvement
- Automatic batch-size selection
- Deterministic 80/20 train/validation split

The best checkpoint is written below
`storage/runs/candy-yolo11s*/weights/best.pt`.

## Foundry Local OCI artifact

The validated ONNX model is packaged as:

```text
storage/artifacts/candy-yolo11s-v1.tar.gz
```

The archive contains exactly one root-level file, `best.onnx`. A local OCI
image layout is available at:

```text
storage/artifacts/candy-yolo11s-oci:candy-yolo11s-v1
```

Verified digests:

```text
ONNX:     sha256:bfe61ab990758f0adbd70d38ec9449555fbe2c62cb1795cc461077a602f760f2
archive:  sha256:0b9d5f289b1aec7669a43ce7c3b190ac9c73e988cc51f166719e33d717253817
manifest: sha256:b485b6d2fe7888174bb1adb9c318da028111d43899f3ab3da857a38dceae19a7
```

Foundry Local requires this archive in an externally addressable
OCI-compatible registry. After authenticating with ORAS, push it with:

```bash
cd ~/SFF-Ignite-Demo/YOLO-training/storage/artifacts
../tools/oras push REGISTRY/REPOSITORY:candy-yolo11s-v1 \
  candy-yolo11s-v1.tar.gz:application/gzip
```

The deployed ACR artifact is:

```text
ignitedemoregistry.azurecr.io/models/candy-yolo11s:v1
```

The Foundry Local inline BYO manifest is:

```text
foundry/modeldeployment.yaml
```

It deploys a GPU ONNX predictive workload through the Jetson NVIDIA runtime
and exposes it through the internal Foundry Gateway as
`candy-yolo11s.local`.

The exported ONNX graph has a fixed batch dimension of one. The manifest sets
`BATCH_SIZE=1`; otherwise the predictive runtime defaults to 32 and pads a
single request to a shape the model rejects.

Deployed endpoint:

```text
https://inference-internal-gateway-istio.foundry-local-operator.svc.cluster.local:443/candy-yolo11s
```

Verified API schema:

```text
input:  images  [1, 3, 640, 640] float32
output: output0 [1, 15, 8400]    float32
```
