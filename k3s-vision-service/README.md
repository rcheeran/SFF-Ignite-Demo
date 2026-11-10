# k3s OWL-ViT vision service

This service is the exclusive owner of the wrist-mounted D405. It provides:

- Aligned 640x480 color/depth capture at 30 FPS.
- `google/owlvit-base-patch32` inference on the Jetson Thor GPU.
- Corrected D405 depth and metric camera-frame XYZ.
- An FFmpeg MJPEG live stream.
- Three annotated object-scan panels and three proxied hand-scan panels.
- An HTTP API consumed by `demo.py`.

This is the alternative **combined OWL-ViT service**, not the independent
[camera service](../k3s-OpenCV-service/README.md) used by the current demo.
Its legacy `/api/capture` endpoint remains available. The current demo instead
orchestrates independent camera and YOLO detection services.

The service does not connect to the Piper CAN interface and does not transform
points into the Piper base frame. `demo.py` combines returned camera XYZ with
the robot pose measured around each HTTP capture.

## Deploy

```bash
cd ~/SFF-Ignite-Demo
./k3s-vision-service/build-and-deploy.sh
```

The script builds the ARM64 CUDA image with host networking, imports it into
k3s containerd, applies `k8s.yaml`, and waits for rollout.

Open:

```text
http://10.0.0.19:8094/
```

The host firewall permits this port only from `10.0.0.0/24`.

## Verify

```bash
curl http://127.0.0.1:8094/healthz
sudo kubectl -n productdaydemo get pods
```

Expected health:

```json
{
  "status": "ok",
  "model": "google/owlvit-base-patch32",
  "camera_serial": "126122270802",
  "cuda": true
}
```

Manual capture:

```bash
curl -X POST http://127.0.0.1:8094/api/capture \
  -H 'Content-Type: application/json' \
  -d '{"slot":1,"prompt":"red cube","confidence":0.2}'
```

No-robot prompt and threshold sweep:

```bash
cd ~/SFF-Ignite-Demo
python Demo-Product-Day/test_owlvit.py "red cube" --legacy-capture
```

This tests thresholds `0.20`, `0.10`, and `0.05` by default and prints the
score, box, camera coordinates, and annotated snapshot URL for each detection.
Override them with, for example, `--thresholds 0.1 0.05 0.03`. This command
does not connect to or move the Piper.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Live stream and three scan panels |
| `GET` | `/healthz` | Model, CUDA, and camera health |
| `GET` | `/stream.mjpg` | FFmpeg MJPEG stream |
| `GET` | `/api/snapshots` | Latest metadata for slots 1-3 |
| `GET` | `/snapshots/1.jpg` | Annotated slot image; likewise 2 and 3 |
| `POST` | `/api/capture` | Capture and infer one aligned frame |

Internal synchronized hand-service endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/rgbd-capture` | Store an aligned RGB/depth pair and return a token |
| `GET` | `/rgbd/{token}.jpg` | RGB image associated with that token |
| `POST` | `/api/deproject` | Deproject a pixel using that token's exact depth |

The dashboard proxies hand metadata and annotated images through port 8094, so
browser clients do not need direct cross-origin access to port 8095.

While `demo.py` supplies fresh robot telemetry, the live stream and object
snapshots show the configured object workspace as a green perspective
wireframe. The projection uses the live base-to-camera transform, calibrated
hand-eye matrix, and D405 intrinsics. It expires after five seconds so a stale
box is not displayed after untracked arm motion.

Capture request:

```json
{
  "slot": 1,
  "prompt": "red cube",
  "confidence": 0.2
}
```

The response status is `detected`, `not_found`, or `rejected`. A detected
response includes bounding box, selected pixel, raw/corrected depth, and
camera-frame XYZ in millimetres.

## Operations

Restart:

```bash
sudo kubectl -n productdaydemo rollout restart deployment/owlvit-vision
sudo kubectl -n productdaydemo rollout status deployment/owlvit-vision
```

Stop camera ownership before running a host calibration tool:

```bash
sudo kubectl -n productdaydemo scale deployment/owlvit-vision --replicas=0
```

Restart afterward:

```bash
sudo kubectl -n productdaydemo scale deployment/owlvit-vision --replicas=1
sudo kubectl -n productdaydemo rollout status deployment/owlvit-vision
```

Only one process may own the D405 at a time.
