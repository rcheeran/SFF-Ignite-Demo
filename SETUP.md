# Piper Arm Setup

Steps to get a Piper arm from a fresh Ubuntu machine to a verified, controllable
CAN connection. Run `bash setup.sh` first (system deps, venv, CAN activation -
see README.md), then follow the steps below, in order.

## 1. Put the arm in Slave mode

1. **Turn off the button on Joint 5.** If this button is on/engaged, the arm
   stays in Master (leader) mode - meant for manual teach/drag control, not
   CAN scripted control. In Master mode the arm's status/joint-angle telemetry
   (CAN IDs 0x2A1-0x2A8) does not broadcast, and motion commands sent over CAN
   are accepted onto the bus but not executed.
2. Set Slave mode over CAN:

   ```python
   from piper_sdk import *
   piper = C_PiperInterface_V2("can0")
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
bash scripts/can_activate.sh can0 1000000
```

Verify:

```bash
ip -details link show can0   # expect: state UP, can state ERROR-ACTIVE, bitrate 1000000
```

## 4. Verify the connection

```bash
source ~/venv/bin/activate
python3 scripts/check_connection.py
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
