# k3s OpenCV + Foundry YOLO vision service

This service is the exclusive owner of the wrist-mounted Intel RealSense D405.
It replaces the OWL-ViT service on port 8094 while preserving the API consumed
by `demo.py` and the hand service.

Components:

- Aligned 640x480 D405 color/depth capture at 30 FPS.
- OpenCV letterbox preprocessing and non-maximum suppression.
- Candy YOLO11s inference through the Foundry Local GPU endpoint.
- Starburst long-axis estimation with a magenta snapshot overlay.
- Independent depth, camera XYZ, and orientation for every detected Starburst.
- Corrected D405 depth and metric camera-frame XYZ.
- FFmpeg MJPEG live stream and three annotated object scan panels.
- Proxied hand scan panels and synchronized RGB-D hand endpoints.
- Live projected Piper object-workspace overlay.
- Durable raw Starburst images and YOLO labels for model improvement.

Supported object prompts include `starburst`, `starburst candy`, `snickers`,
`skittles`, `nerds`, `twizzlers`, `airheads`, `gummy worms`, `milky way`,
`three musketeers`, and the two M&M classes.

Starburst pickup is rejected when the wrapper contour is not sufficiently
elongated to determine a safe orientation. `demo.py` transforms the accepted
axis through hand-eye calibration, compares its undirected angle between scan
views, and aligns the gripper fingers with the candy length.

The capture response preserves the highest-confidence valid detection in the
top-level fields and returns every YOLO result in `candidates`. The annotated
snapshot numbers valid candies as `#1`, `#2`, and so on.

Every Starburst capture containing at least one detection also saves:

```text
datasets/collected-starburst/images/<unique-id>.jpg
datasets/collected-starburst/labels/<unique-id>.txt
```

Images are the unannotated 640x480 camera frames. Each matching label file uses
normalized YOLO format with Starburst class ID 8:

```text
8 center_x center_y width height
```

Multiple Starbursts produce multiple lines in one label file. Captures with no
detections are not saved because an empty label would incorrectly identify a
missed Starburst as a negative training image. The labels are detector-generated
pseudo-labels and should be reviewed before adding them to a training split.
The files persist across pod restarts and OpenCV service rollouts.

## Deploy

The OWL-ViT deployment must remain scaled to zero because only one process can
own the D405.

```bash
cd ~/productdaydemo
./OpenCV-service/build-and-deploy.sh
```

The build script copies the operator-generated Foundry API key and CA
certificate into namespace-local Kubernetes resources without printing the
key.

Open:

```text
http://10.0.0.19:8094/
```

## Verify

```bash
curl http://127.0.0.1:8094/healthz

curl -X POST http://127.0.0.1:8094/api/capture \
  -H 'Content-Type: application/json' \
  -d '{"slot":1,"prompt":"starburst candy","confidence":0.25}'
```

The response contract and synchronized hand endpoints match
`vision-service/README.md`.

## Operations

```bash
sudo kubectl -n productdaydemo rollout restart deployment/opencv-vision
sudo kubectl -n productdaydemo scale deployment/opencv-vision --replicas=0
sudo kubectl -n productdaydemo scale deployment/opencv-vision --replicas=1
```
