# k3s MediaPipe Hand Landmarker service

This service consumes the OpenCV service's D405 MJPEG stream and runs
MediaPipe Hand Landmarker without opening the camera device. This preserves
exclusive D405 ownership in `opencv-vision`.

## Deploy

```bash
cd ~/SFF-Ignite-Demo
chmod +x k3s-hand-service/build-and-deploy.sh
./k3s-hand-service/build-and-deploy.sh
```

Open:

```text
http://10.0.0.19:8095/
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Annotated stream and latest JSON |
| `GET` | `/healthz` | Source/model/service health |
| `GET` | `/api/hands` | Latest handedness, 21 landmarks, and palm center |
| `GET` | `/api/snapshots` | Hand scan metadata for slots 4-6 |
| `POST` | `/api/capture` | Stable synchronized RGB-D hand capture for slot 4-6 |
| `GET` | `/snapshots/{4,5,6}.jpg` | Annotated hand scan image |
| `GET` | `/frame.jpg` | Latest annotated JPEG |
| `GET` | `/stream.mjpg` | Annotated MJPEG stream |

Landmarks are normalized image coordinates. MediaPipe world landmarks are
reported in metres relative to the hand model, not in the D405 or Piper base
frame. Slot capture requests use a tokenized aligned RGB-D frame from the
camera-owning service and deproject the palm pixel against that exact depth
frame. The response includes D405 camera-frame XYZ in millimetres.
The palm target is the midpoint between the wrist landmark and the centroid of
the four finger-base knuckles, rather than an equal average that overweights
the finger side of the palm.

Test without moving the robot by holding a palm in the current D405 view:

```bash
curl -X POST http://127.0.0.1:8095/api/capture \
  -H 'Content-Type: application/json' \
  -d '{"slot":4}'
```

Five samples are taken; at least three must contain an open palm with all four
non-thumb fingers straight and extended. Palm coordinates within 50 mm of the
five-sample median form the stable cluster; one or two positional outliers are
discarded, and at least three stable samples are required. The final coordinate
is the median of that stable cluster. A closed or partially closed hand is
reported in the live stream but is not accepted as a delivery target. A
synchronized frame with unusable palm depth is rejected as an individual
sample rather than aborting the slot; at least three samples must still have
valid depth. The capture response reports visible-hand samples, open-palm
detections, valid depth samples, stable and outlier counts, stable and raw
spread, and exact depth-rejection reasons.

## Operations

```bash
curl http://127.0.0.1:8095/healthz
curl http://127.0.0.1:8095/api/hands
sudo kubectl -n productdaydemo get pods
sudo kubectl -n productdaydemo logs deployment/hand-landmarker
```

The service reconnects if the OpenCV MJPEG stream restarts. It does not require
GPU access, privileged mode, or a USB device mount.
