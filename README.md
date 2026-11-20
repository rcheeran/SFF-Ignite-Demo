# SFF AgileX Piper

Setup, calibration, vision services, and demo workflows for the AgileX Piper.

## Repository layout

- [prepare-jetson](prepare-jetson/README.md): Jetson setup and CAN recovery.
- [prepare-piper](prepare-piper/): robot connectivity and camera calibration.
- [Demo-Ignite](Demo-Ignite/README.md): supervised D435 Starburst pickup and fresh-palm delivery loop
  ([event-day checklist](Demo-Ignite/EVENT_DAY.md)).
- [k3s-OpenCV-service](k3s-OpenCV-service/README.md): camera, depth, and public vision API.
- [k3s-vision-detection](k3s-vision-detection/README.md): RGB processing and Foundry YOLO detection.
- [k3s-hand-service](k3s-hand-service/README.md): MediaPipe hand service.
- [k3s-vision-service](k3s-vision-service/README.md): alternative OWL-ViT vision service (not deployed).
- [YOLO-training](YOLO-training/README.md): candy-model training.
- [useful-piper-scripts](useful-piper-scripts/): standalone diagnostics and utilities.

Start with the [setup and supervised demo instructions](prepare-jetson/README.md).
Commands there run from the repository root unless stated otherwise.
