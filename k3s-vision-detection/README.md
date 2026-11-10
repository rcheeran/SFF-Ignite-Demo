# RGB vision detection service

HTTP service on port 8096. It owns RGB letterboxing, Foundry YOLO11s
inference, class filtering, NMS, and wrapper-contour orientation estimation.
Inference still runs on the existing Foundry GPU service; this container needs
no local GPU, RealSense, USB access, privileged mode, or calibration files.

```text
Demo -> Camera :8094: synchronized capture, token, lossless PNG
Demo -> RGB detection :8096 -> Foundry candy-yolo11s: boxes and pixel axes
Demo -> Camera :8094: localize boxes/axes using that token's depth
```

The [camera service](../k3s-OpenCV-service/README.md) exclusively owns USB and
provides capture, depth/XYZ, streaming, and snapshots. The hand service still
connects to port 8094. The camera never calls detection.

[detection_client.py](detection_client.py) is the application's HTTP adapter.
The demo's [vision_client.py](../Demo-Product-Day/vision_client.py) coordinates
both services. Its `--detection-url` defaults to `http://127.0.0.1:8096`.

## Internal API

- `GET /healthz`: checks Foundry and returns
  `{"status":"ok","model":"candy-yolo11s","cuda":true}`; unavailable returns 503.
- `POST /api/detect`: JSON object with `prompt`, `confidence` (strictly between
  zero and one), and `image_png_base64` (base64-encoded lossless PNG).
  Response: `{"status":"ok","model":"candy-yolo11s","detections":[...]}`.
  Detection dictionaries preserve `score`, `box_xyxy`, `long_axis_pixels`,
  orientation quality fields, or `orientation_error`. An empty list means no
  matching candy, not an inference failure.

Bodies are limited to 8 MiB and images to 2,073,600 pixels (1920x1080 area).
Invalid input returns 400, an oversized body 413, concurrent inference 409,
and Foundry failures 502, all with explicit JSON errors.
Only one inference request runs at a time. PNG transport preserves colors and
pixels used for orientation; no JPEG recompression is introduced.

The service provides a ClusterIP endpoint and a host port bound to **127.0.0.1**
for a demo running directly on the Jetson. It does not use host networking.
Recreate rollouts avoid conflicting host-port allocations. For a remote demo,
use a trusted tunnel or explicitly configure routing; the host port is not
exposed on the LAN. This is not a public/authenticated upload endpoint.

## Configuration and deployment

`DETECTION_MODEL` defaults to `candy-yolo11s`; `PORT` defaults to 8096.
Foundry settings moved here from the camera deployment:

- `FOUNDRY_URL`: the candy model's HTTPS endpoint.
- `FOUNDRY_API_KEY`: required; supplied by the namespace-local secret.
- `FOUNDRY_CA_CERT`: trusted Foundry CA file.
- `FOUNDRY_TIMEOUT_SECONDS`: default 30.
- `NMS_IOU_THRESHOLD`: default 0.45.

The model must be loaded with input shape `[1,3,640,640]` and
`CUDAExecutionProvider`. Supported candy prompts and results are unchanged.

When ready to deploy, from the repository root:

```bash
bash k3s-vision-detection/build-and-deploy.sh
bash k3s-OpenCV-service/build-and-deploy.sh
```

Neither service requires the other to start. The detection script creates the namespace before copying the Foundry operator's
API key and CA into namespace-local resources, builds/tests/imports the image,
and rolls out detection. The camera script separately deploys the camera/API.
Only the demo requires both healthy services before it can begin scanning.
These scripts change running services; do not run them for offline validation.

## Hardware-free validation

With the pinned requirements installed in an isolated Python environment:

```bash
python -m unittest discover -s k3s-vision-detection -v
```

The source-tree integration test runs the real application orchestrator against
temporary camera and detection HTTP servers. It mocks camera hardware and
Foundry, checking exact tensor parity, multiple candies, ambiguous orientation,
and localization against the original depth even after the live frame changes.
It also verifies that detector failure does not break camera health and that
expired tokens fail rather than silently recapturing. It requires neighboring
source folders and `pyrealsense2` (used only for deprojection, not hardware).

The container includes detector/API unit tests, without source-tree integration:

```bash
sudo docker build -t productdaydemo/vision-detection:latest k3s-vision-detection
sudo docker run --rm productdaydemo/vision-detection:latest \
  python -m unittest discover -s /app/tests -v
```
