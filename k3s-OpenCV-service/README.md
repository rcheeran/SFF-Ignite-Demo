# Independent camera and RGB-D service

This service exclusively owns the wrist-mounted RealSense camera. It captures
aligned color/depth, streams MJPEG, corrects depth, deprojects pixels/boxes into
camera XYZ, and stores annotated scan snapshots and dataset images.

**It does not call detection, load a model, or require detector/GPU health.**
The main application coordinates the services:

```text
demo.py / vision_client.py
  1. Camera :8094        -> capture RGB-D, receive token and PNG URL
  2. Detection :8096     -> detect candies in that exact PNG
  3. Camera :8094        -> localize returned boxes/axes using the same token
  4. Demo               -> transform camera XYZ and plan robot actions
```

The application-side [orchestrator](../Demo-Product-Day/vision_client.py) uses
the [detection client](../k3s-vision-detection/detection_client.py).
The camera has no detection URL, credentials, or client dependency.

The video widget title identifies the device-reported camera model and owning
service (`opencv-vision`). There is no separate Camera Information panel or
legacy D405/OWL-ViT banner painted over the video. `/healthz` retains the full
identity as `camera_name`, `camera_serial`, and `camera_owner_service`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Camera health and capabilities, without model/CUDA claims |
| GET | `/` | Dashboard: live stream beside the latest candy and hand scans, scan details below |
| GET | `/stream.mjpg` | Live MJPEG stream |
| POST | `/api/rgbd-capture` | Capture one aligned RGB/depth pair and return its token |
| GET | `/rgbd/{token}.png` | Lossless RGB image for detection |
| GET | `/rgbd/{token}.jpg` | Existing JPEG route used by the hand service |
| POST | `/api/localize` | Localize caller-supplied detections against token-bound depth |
| POST | `/api/deproject` | Existing token-bound single-pixel hand deprojection |
| POST | `/api/workspace-overlay` | Update the projected robot workspace |
| GET | `/api/snapshots` | Latest caller-published object scan metadata |
| GET | `/snapshots/{slot}.jpg` | Annotated object scan image |

`POST /api/capture` intentionally returns **410 Gone** with migration guidance.
It no longer triggers camera-side inference. Update callers to orchestrate the
three steps above; no implicit legacy fallback is used.

Capture response example:

```json
{
  "status": "ok",
  "token": "<frame-token>",
  "frame_sequence": 123,
  "captured_at": "<UTC timestamp>",
  "width": 640,
  "height": 480,
  "image_url": "/rgbd/<frame-token>.jpg",
  "image_png_url": "/rgbd/<frame-token>.png"
}
```

Localization request example:

```json
{
  "token": "<frame-token>",
  "slot": 1,
  "prompt": "starburst candy",
  "model": "candy-yolo11s",
  "detections": [
    {"score": 0.9, "box_xyxy": [120, 140, 240, 200]}
  ]
}
```

Include detector-returned orientation fields unchanged when available.
Ignite also adds `"pickup_point": "long_axis_midpoint"` to each detection.
This requires valid `long_axis_pixels` endpoints and deprojects their arithmetic
image midpoint (including fractional pixels) using the existing corrected
box-filtered depth. Missing/degenerate endpoints reject that candidate without
falling back to a box center. Unknown pickup modes also reject the candidate.
The response echoes `pickup_point`, uses the selected point for `pixel_uv` and
`camera_point_mm`, and reports the original depth-cluster pixel median separately
as `depth_sample_pixel_uv`. Snapshot crosses mark the pickup point.
Omitting `pickup_point` retains the existing `depth_cluster` localization mode
for older callers. This does not change hand deprojection or depth filtering.
`model` and `prompt` are caller metadata, not camera-side model selection.
The response retains the existing detection/candidate counts, per-candidate
status and depth diagnostics, highest-confidence valid candidate, corrected
camera XYZ, 3D long-axis geometry, snapshot URL, and dataset paths. It also
returns the token and original capture time/sequence.

Invalid/ambiguous orientations remain rejected candidates. No detections returns
`not_found`. Bad geometry or an unknown/evicted token returns an explicit 400;
concurrent localization returns 409. Localization never grabs a replacement
frame. The cache holds the latest **30 captures**; callers must finish before
their token is evicted. Localization bodies are limited to 2 MiB / 1000 boxes.

The hand service's JPEG, RGB-D, deprojection, and MJPEG contracts are unchanged.

The dashboard shows the live stream (about 70% of the page width, up to
1113 x 835 px on a 1920 px screen) beside the latest candy scan stacked above
the latest hand scan (up to 460 x 345 px each). The candy and hand scan
metadata appear side by side below. When several scan poses publish
snapshots, the most recently captured candy and hand scans are shown. Below
900 px of window width everything stacks in one column. The candy panel
updates when the demo or threshold tester publishes localized results, not
through background inference.

## Dataset collection

When enabled, localization saves the original captured image and provisional
YOLO labels for the configured prompts:

```text
datasets/collected-starburst/images/<unique-id>.jpg
datasets/collected-starburst/labels/<unique-id>.txt
```

Starburst labels retain class ID 8. Multiple boxes produce multiple lines;
empty detection sets are not saved as misleading negative samples. Review
these pseudo-labels before training.

## Deployment and use

Camera and detection may be deployed independently. When ready, from the
repository root:

```bash
bash k3s-OpenCV-service/build-and-deploy.sh
bash k3s-vision-detection/build-and-deploy.sh
```

The alternative OWL-ViT deployment has been removed from the cluster; do not
redeploy it alongside this service, because only one process can own the USB
camera. The camera script no longer provisions Foundry credentials.
The detection manifest maps port 8096 to Jetson loopback for local demo callers,
while also providing an internal ClusterIP service.

The camera manifest selects the connected **D435, serial 263022078260**, and
mounts `prepare-piper/hand-eye-realsense435/calibration`. It currently sets
`CAMERA_PREVIEW_ONLY=false` to use the accepted D435 depth correction.

Optional preview-only mode permits streaming and token-bound JPEG/PNG image capture.
Health reports `preview_only: true`, `depth_calibrated: false`, and only the
`rgbd-capture` capability. The dashboard displays a warning. Depth/XYZ
localization, pixel deprojection, and workspace overlays explicitly reject
requests; no old or identity depth correction is substituted.

Missing/rejected depth calibration prevents calibrated-mode startup. Full robot
operation also requires accepted D435 hand-eye calibration and reviewed demo
configuration. [Ignite](../Demo-Ignite/README.md) defaults to D435 artifacts;
the older Product Day entry point defaults to the moved
`hand-eye-realsense405` directory. Do not reuse its old extrinsics.

Hardware-free/unit tests in the camera image:

```bash
sudo docker build -f k3s-OpenCV-service/Dockerfile \
  -t productdaydemo/opencv-vision:latest .
sudo docker run --rm productdaydemo/opencv-vision:latest \
  python -m unittest discover -s /app/tests -v
```

After calibration and enabling metric localization, a no-robot threshold test:

```bash
python Demo-Product-Day/test_owlvit.py "starburst candy" \
  --vision-url http://127.0.0.1:8094 \
  --detection-url http://127.0.0.1:8096
```

This does not import robot control or command arm motion.
