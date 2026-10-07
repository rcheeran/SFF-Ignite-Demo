# Piper Arm Setup

Steps to get a Piper arm from a fresh Ubuntu machine to a verified, controllable
CAN connection. Run `bash prepare-jetson/setup.sh` first (system deps, venv, CAN activation -
see README.md), then follow the steps below, in order.

## Host packages and camera preview access

`prepare-jetson/setup.sh` installs the required Python, CAN, USB, RealSense,
build, and video packages. The video packages include `ffmpeg`, which the
fixed-pose homography workflow uses to encode the wrist-camera alignment
preview as MJPEG.

The setup also adds the operator to `video`, `plugdev`, and `render`. Log out
and back in, reboot, or reconnect the VS Code remote session before using the
D405 or CUDA. Verify the effective groups with:

```bash
id
```

OWL-ViT additionally requires:

```bash
source ~/venv/bin/activate
pip install -r prepare-jetson/requirements-demo.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected on the Jetson Thor: `True NVIDIA Thor`.

Verify FFmpeg after setup:

```bash
ffmpeg -version | head -n 1
```

The homography alignment page listens on TCP port `8094`. This Jetson's k3s
firewall has a default-drop input policy, so allow that port only from the
local `10.0.0.0/24` network:

```bash
sudo iptables -C INPUT -p tcp -s 10.0.0.0/24 --dport 8094 -j ACCEPT 2>/dev/null ||
  sudo iptables -I INPUT 1 -p tcp -s 10.0.0.0/24 --dport 8094 -j ACCEPT
```

Verify the scoped rule:

```bash
sudo iptables -C INPUT -p tcp -s 10.0.0.0/24 --dport 8094 -j ACCEPT
sudo iptables -L INPUT -n --line-numbers | head
```

This does not expose the preview outside the local subnet. The runtime rule
may be removed by a reboot or a k3s firewall reload; rerun the idempotent
command above if the preview URL times out. To remove the rule manually:

```bash
sudo iptables -D INPUT -p tcp -s 10.0.0.0/24 --dport 8094 -j ACCEPT
```

While `record_scan_pose.py` is running, open:

```text
http://<jetson-ip>:8094/
```

For this machine the current address is `http://10.0.0.19:8094/`. The server
stops when the script exits, so the URL is expected to refuse connections at
other times.

## 1. Put the arm in Slave mode

1. **Turn off the button on Joint 5.** If this button is on/engaged, the arm
   stays in Master (leader) mode - meant for manual teach/drag control, not
   CAN scripted control. In Master mode the arm's status/joint-angle telemetry
   (CAN IDs 0x2A1-0x2A8) does not broadcast, and motion commands sent over CAN
   are accepted onto the bus but not executed.
2. Set Slave mode over CAN:

   ```python
   from piper_sdk import *
   piper = C_PiperInterface_V2("can_piper")
   piper.ConnectPort()
   piper.MasterSlaveConfig(0xFC, 0, 0, 0)   # 0xFA = Master, 0xFC = Slave
   ```

   (This matches AgileX's own `piper_sdk/demo/V2/piper_set_slave.py`.)

## 2. Restart the arm

**Required if the arm was previously in Master mode** - the Master/Slave
config change does not take effect until the arm is power-cycled. Skipping
this step leaves the arm stuck reporting `Control Mode: TEACHING_MODE` /
`Teach Status: START_RECORDING` even after the Slave config command succeeds.

## 3. Reactivate the CAN interface

The USB-CAN adapter drops its interface state across an arm power cycle
(comes back up as `DOWN`/`STOPPED`, sometimes under a new interface index).
Bring it back up:

```bash
sudo systemctl restart piper-can.service
```

Verify:

```bash
ip -details link show can_piper   # expect: state UP, can state ERROR-ACTIVE, bitrate 1000000
```

## 4. Verify the connection

```bash
/home/clouduser/venv/bin/python prepare-piper/check_connection.py can_piper
```

Expect:
- `fps` around 3000-3100 (roughly half that, ~1400-1500, means the arm is
  still broadcasting only low-level per-joint driver frames and not its
  high-level status range - go back to step 1)
- `ArmStatus.Hz` and `JointMsgs.Hz` both non-zero, with real values
- `Control Mode: STANDBY(0x0)` (or `CAN_CTRL(0x1)` if already switched)
- `EnableStatus` all `True` once enabled

## 5. Enable and switch to CAN_CTRL mode

```python
piper.EnablePiper()
piper.ModeCtrl(0x01, 0x01, 30, 0x00)   # ctrl_mode=CAN_CTRL, move_mode=MOVE_J, speed=30%
```

Re-check `piper.GetArmStatus()` - `Control Mode` should now read `CAN_CTRL(0x1)`.
The arm is ready for scripted motion commands at this point.

## Notes

- If the CAN adapter re-enumerates (e.g. after any arm power cycle), repeat
  step 3 before doing anything else - commands sent while the interface is
  down fail with `SendCanMessage(SEND_MESSAGE_FAILED ...)`.
- If you reconnect (`ConnectPort()`) in a *new* Python process after the CAN
  interface was brought down and back up, use a fresh `C_PiperInterface_V2`
  instance rather than reusing one created before the interface dropped - the
  old instance's underlying socket goes stale.
