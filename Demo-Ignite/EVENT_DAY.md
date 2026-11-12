# Ignite demo: event-day checklist

> **Automatic start is enabled.** About a minute after the Jetson boots, the
> `ignite-demo` service starts the demo and **the arm moves on its own**. Keep the
> area clear and the emergency stop reachable whenever the Jetson powers on. To
> run the checks below or start the demo by hand instead, stop it first:
> `sudo systemctl stop ignite-demo` (see section 8).

Work through the sections in order. Steps 1-7 never move the arm. Run every
command from the repository root with the demo environment active:

```bash
source ~/venv-ignite-demo/bin/activate
cd ~/SFF-Ignite-Demo
```

`kubectl` needs `sudo` on this Jetson (the k3s kubeconfig is root-only).

## 1. Power on and connect

1. Plug the D435 into a USB 3 port and the Piper USB-CAN adapter into USB.
2. Power on the Jetson. The encrypted data disks unlock automatically through
   the TPM; no passphrase is needed.
3. Power on the Piper controller. Keep the emergency stop within reach from
   now on.
4. Wait about two minutes for k3s and the services to start.

## 2. Host checks

```bash
# Repository mount on the encrypted disk (SOURCE must be k3s-data-secure)
findmnt -T ~/SFF-Ignite-Demo -o TARGET,SOURCE

# k3s and the CAN bring-up service (both should print "active")
systemctl is-active k3s piper-can

# Both USB devices are visible
lsusb | grep -E "RealSense D435|CAN adapter"
```

Expected `lsusb` lines:

```text
... ID 8086:0b07 Intel Corp. RealSense D435
... ID 1d50:606f OpenMoko, Inc. Geschwister Schneider CAN adapter
```

## 3. CAN link and arm

```bash
ip -details link show can_piper | grep "can state"
```

`can state ERROR-ACTIVE` is healthy. `ERROR-PASSIVE` or `BUS-OFF` means the
arm is not answering (powered off, emergency stop pressed, or the adapter was
re-plugged). Reset the bus with:

```bash
bash prepare-jetson/can_up.sh
```

Then run the read-only arm check (no motion is sent):

```bash
python prepare-piper/check_connection.py can_piper
```

Healthy output includes:

| Line | Expected |
|---|---|
| `ArmStatus.Hz` / `JointMsgs.Hz` | about `200.0` (`0.0` means the arm is in Master mode or needs a power cycle; see [SETUP.md](../prepare-jetson/SETUP.md)) |
| `Control Mode` | `CAN_CTRL` |
| `Arm Status` | `NORMAL` |
| `Teach Status` | `STOP_RECORDING` |
| `Error Code` | `0` |
| `Joint N Angle Limit Status` / `Communication Status` | all `False` (no fault) |
| `EnableStatus` | six `True` values (may be `False` after a fresh power-on; the demo enables the arm) |

Only one program can use the CAN bus safely at a time: do not run this check
while `demo.py` is running.

## 4. Kubernetes pods and services

```bash
# Node ready
sudo kubectl get nodes

# Demo services: hand-landmarker, opencv-vision, vision-detection must be 1/1 Running
sudo kubectl -n productdaydemo get deploy,pods,svc

# Foundry Local YOLO model used by vision-detection
sudo kubectl -n foundry-local-operator get modeldeployments
sudo kubectl -n foundry-local-operator get pods -l app.kubernetes.io/name=candy-yolo11s

# Anything not Running or Completed across the cluster
sudo kubectl get pods -A | grep -vE "Running|Completed"
```

What good looks like:

- `productdaydemo`: `hand-landmarker`, `opencv-vision`, and `vision-detection`
  are `1/1` and `Running`.
- `candy-yolo11s` model deployment: `STATE Running`, `READY true`, and at least
  one `candy-yolo11s-...` pod `2/2 Running`. Older pods from previous boots may
  show `Unknown` or `Error`; they are harmless while one pod is `2/2 Running`.
  To clear them, delete only the stale pods by name (never the `2/2 Running` one):

  ```bash
  sudo kubectl -n foundry-local-operator delete pod <stale-pod-name> [<stale-pod-name> ...]
  ```
- A high `RESTARTS` count is historical if the restart time in brackets is days
  ago. A count that keeps rising means the pod is crash-looping.

Inspect or restart a service:

```bash
sudo kubectl -n productdaydemo logs deploy/opencv-vision --tail=50
sudo kubectl -n productdaydemo rollout restart deploy/opencv-vision
sudo kubectl -n productdaydemo rollout status deploy/opencv-vision --timeout=5m
```

Use the same commands with `hand-landmarker` or `vision-detection`. Restart
`hand-landmarker` after `opencv-vision`, because it reads the camera stream from
it. Restarting `opencv-vision` while the demo runs interrupts the demo.

## 5. Service health

```bash
for port in 8094 8095 8096; do
  echo "$port: $(curl -s -m 10 http://127.0.0.1:$port/healthz)"
done
```

| Port | Service | Expected |
|---|---|---|
| 8094 | Camera (opencv-vision) | `"status": "ok"`, `"camera_name": "RealSense D435"`, `"camera_serial": "263022078260"`, `"depth_calibrated": true`, `"preview_only": false` |
| 8095 | Hand (MediaPipe) | `"status": "ok"`, `"source_url": "http://127.0.0.1:8094/stream.mjpg"` |
| 8096 | YOLO detection | `"status": "ok"`, `"model": "candy-yolo11s"`, `"cuda": true` |

Live views for the audience screen (find the address with `hostname -I`; it
changes with the network, so check it on the event network):

- Camera, detections, and workspace outline: `http://<jetson-ip>:8094/`
- Hand detection: `http://<jetson-ip>:8095/`

## 6. Demo preflight (no CAN, no motion)

```bash
python Demo-Ignite/demo.py --check-only --detector opencv
python Demo-Ignite/demo.py --check-only --detector yolo
```

Each must end with `Preflight passed; no CAN connection or arm motion`. Any
`[FAILED]` line names the service to fix in sections 4-5.

## 7. Physical setup

- Board and table in the taught position; nothing has moved since calibration.
- Clear space around the arm, the scan pose, and the path to the palm area.
- Candies flat on the table, inside the workspace outline shown on the 8094
  view and within about 42 cm (X 420 mm) of the robot base. Further out, candies
  lying along X are grasped at a forward tilt, which missed often.
- The open palm goes on "Place hand here", away from the candies.
- Gripper empty. Emergency stop reachable by the operator.

## 8. Run the demo

### Automatic start at boot

The `ignite-demo` systemd service ([ignite-demo.service](ignite-demo.service))
runs [run_demo_at_boot.sh](run_demo_at_boot.sh) at every boot. It waits until the
CAN link is up, the camera (8094) and hand (8095) services are healthy, YOLO
detection (8096) is healthy (or 5 minutes have passed, since OpenCV mode does not
need it), and the no-motion preflight passes, then starts:
`demo.py --allow-motion --detector opencv --verbose --max-cycles 0`. If anything
is still not ready after 15 minutes it gives up without moving. It does not
restart after an error. Output goes to `Demo-Ignite/logs/run-<timestamp>.log`
and the journal.

```bash
systemctl status ignite-demo           # waiting, running, or stopped
journalctl -u ignite-demo -f           # live output
sudo systemctl stop ignite-demo        # same as Ctrl+C: arm returns to zero, gripper opens
sudo systemctl start ignite-demo       # start again (waits, then the arm moves)
sudo systemctl disable ignite-demo     # turn off automatic start at boot
sudo systemctl enable ignite-demo      # turn it back on
Demo-Ignite/run_demo_at_boot.sh --wait-only   # readiness checks only, no motion
```

Stop the service before running the demo by hand or `check_connection.py`: only
one program may control the arm. The CAN link also comes up automatically
whenever the USB-CAN adapter appears (late boot, arm power cycle, or replug),
via a udev rule installed by [setup.sh](../prepare-jetson/setup.sh).

To install the service on a freshly set-up Jetson:

```bash
sudo install -m 0644 Demo-Ignite/ignite-demo.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable ignite-demo
```

### Manual start

```bash
python -u Demo-Ignite/demo.py --allow-motion \
  --detector opencv --verbose --max-cycles 0 2>&1 \
  | tee -i Demo-Ignite/logs/run-$(date +%Y%m%d-%H%M%S).log
```

- `--allow-motion` is the only motion gate; add it only after the physical
  setup in section 7 is checked. Without it the demo runs the preflight only.
- `--max-cycles 0` loops until Ctrl+C; use a number for a bounded run.
- Drop `--detector opencv` to use YOLO instead. OpenCV is recommended: YOLO can
  miss some wrappers and has mistaken palms for candy.
- On start the arm moves to SDK zero, opens the gripper, then goes to the scan
  pose. Each cycle scans, picks one candy, waits for an open palm, and releases
  76.2 mm above it.

## 9. Stop and shut down

1. Press Ctrl+C once (or `sudo systemctl stop ignite-demo` for the automatic
   start). The arm returns to SDK zero, opens the gripper, and disconnects.
   Wait for it to finish before touching the arm.
2. If anything looks wrong, press the emergency stop instead.
3. Power off the Piper controller, then shut down the Jetson
   (`sudo shutdown -h now`).

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `can state ERROR-PASSIVE` / `BUS-OFF` | Arm off, emergency stop pressed, or adapter re-plugged. Release the stop, power the arm, run `bash prepare-jetson/can_up.sh`. |
| `ArmStatus.Hz: 0.0` | Arm in Master mode or needs a power cycle; see [SETUP.md](../prepare-jetson/SETUP.md). |
| Preflight `[FAILED] OpenCV camera` | Re-seat the D435 on USB 3, then restart `opencv-vision`, then `hand-landmarker` (section 4). |
| Preflight `[FAILED] YOLO / Foundry detection` | Check the `candy-yolo11s` model deployment (section 4), restart `vision-detection`, or run with `--detector opencv`. |
| `vision-detection` in `CrashLoopBackOff` with `Foundry response must be a JSON object` | The model pod is up without a loaded model. Restart it, wait for `2/2 Running`, then restart detection: `sudo kubectl -n foundry-local-operator delete pod -l app.kubernetes.io/name=candy-yolo11s`, then `sudo kubectl -n productdaydemo rollout restart deploy/vision-detection`. The model loads from the on-device store, so this works offline. |
| Demo did not start after boot | `journalctl -u ignite-demo -b` shows what it was waiting for. Fix that item (sections 2-5), then `sudo systemctl start ignite-demo`. |
| `kubectl logs` fails with `proxy error ... localhost:40343, code 502` | k3s routes its own connections through the Azure Arc proxy. Read container logs directly: `sudo k3s crictl ps -a --name detection`, then `sudo k3s crictl logs <container-id>`. |
| Preflight `[FAILED] MediaPipe hand service` | Restart `hand-landmarker` after `opencv-vision` is healthy. |
| `No accepted candy/palm pair` every cycle | A pickup needs a candy and an open palm in the same scan. Check both on the 8094/8095 views; move the candy toward the board centre. |
| `Pickup missed` | The arm rescans and retries automatically. Repeated misses on one candy: reposition it, or check the verbose `PICK PLAN` lines in the log. |
| `No reachable candy` | The candy is too far or at an unreachable angle; move it closer to the robot. |
| `opencv-vision` / `hand-landmarker` restarting, `0/1`, probes report `network is unreachable` | The Jetson's network address changed while k3s was running (new network or DHCP lease), and k3s still uses the old address. Compare `hostname -I` with `sudo kubectl get node -o wide` (INTERNAL-IP). If they differ: `sudo systemctl restart k3s`, wait for the node to show the new address, then `sudo kubectl -n productdaydemo rollout restart deploy/opencv-vision` and, once it is `1/1`, `deploy/hand-landmarker`. Booting on the event network avoids this. |
| Arm faults or joints disabled | Press the emergency stop, power-cycle the arm, run `can_up.sh`, then repeat section 3. |

Keep the run logs in `Demo-Ignite/logs/` for later review.
