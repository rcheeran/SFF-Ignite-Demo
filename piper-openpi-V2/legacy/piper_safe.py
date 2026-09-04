#!/usr/bin/env python3
"""Shared safety layer for all Piper motion scripts.

Rules enforced here, because violating any of them has already cost us hardware
time:

  1. TABLE FLOOR. The end effector is never commanded below Z_FLOOR_MM, and
     every streamed tick re-reads the real EE height - if it drops below
     Z_ABORT_MM the motion aborts and the arm lifts immediately. Predicting is
     not enough: a joint target that *interpolates* to a safe height can still
     dip on the way there.

  2. START FROM ZERO. Every script homes to joint zero before doing anything
     else, so motion always begins from one known, table-clear configuration
     instead of wherever the previous run happened to stop.

  3. WRIST STAYS CAMERA-UP. J6 lives near -165 deg so the wrist-mounted D405
     sits above the gripper. The 180-degree-flipped wrist looks identical by eye
     (the jaws are symmetric) but puts the camera underneath, where a descent
     drives it into the table and loads the flange.

  4. J6 IS THE WEAK MOTOR. Smallest motor, carries gripper + camera. Under
     Cartesian control (which re-solves wrist roll per target) it ran 66-77 C
     and thermally tripped; with the wrist frozen it stays at 37-38 C. Check
     j6_health() between episodes, not mid-motion.
"""
import time

import numpy as np

IFACE = "can_piper"
HZ = 15
DT = 1.0 / HZ
SPEED = 18

# --- table protection -------------------------------------------------------
Z_FLOOR_MM = 100.0     # never command the EE below this
Z_ABORT_MM = 92.0      # measured height that triggers an emergency lift
Z_SAFE_MM = 235.0      # transit height, clear of the can
LIFT_ON_ABORT_MM = 60.0

# --- wrist held camera-up ---------------------------------------------------
J4_HOLD, J5_HOLD, J6_HOLD = 46.0, -23.8, -165.0
J6_LIMIT_MAX, J6_LIMIT_MIN = -1000, -1790     # 0.1 deg: keep the camera up
J6_TEMP_PAUSE, J6_TEMP_RESUME = 46, 41

# HOME = shoulder/elbow at zero (upright, well clear of the table) with the WHOLE
# wrist already in its working configuration. Homing to a different wrist pose
# would swing J4/J5 ~70 deg at the start of every single move for no benefit,
# and J6=0 is 165 deg from camera-up (and blocked by the J6 limit anyway).
HOME = [0.0, 0.0, 0.0, J4_HOLD, J5_HOLD, J6_HOLD]
ZERO = HOME          # backwards-compatible alias


class TableFloorViolation(RuntimeError):
    pass


# ---------------------------------------------------------------- basics
def connect(enable=True):
    from piper_sdk import C_PiperInterface_V2
    p = C_PiperInterface_V2(IFACE)
    p.ConnectPort()
    time.sleep(1.0)
    if enable:
        t0 = time.time()
        while time.time() - t0 < 6:
            p.EnablePiper()
            time.sleep(0.3)
            if all(p.GetArmEnableStatus()):
                break
        if not all(p.GetArmEnableStatus()):
            raise SystemExit(f"ABORT: arm not enabled {p.GetArmEnableStatus()} "
                             "- teach button off? e-stop released?")
    p.MotorAngleLimitMaxSpdSet(motor_num=6, max_angle_limit=J6_LIMIT_MAX,
                               min_angle_limit=J6_LIMIT_MIN, max_joint_spd=0x7FFF)
    return p


def joints(p):
    js = p.GetArmJointMsgs().joint_state
    return np.array([getattr(js, f"joint_{i+1}") / 1000.0 for i in range(6)])


def pose(p):
    ep = p.GetArmEndPoseMsgs().end_pose
    return np.array([ep.X_axis, ep.Y_axis, ep.Z_axis,
                     ep.RX_axis, ep.RY_axis, ep.RZ_axis]) / 1000.0


def ee_z(p):
    return p.GetArmEndPoseMsgs().end_pose.Z_axis / 1000.0


def grip_mm(p):
    return p.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0


def j6_health(p):
    m = p.GetArmLowSpdInfoMsgs().motor_6
    f = m.foc_status
    ok = not (f.driver_error_status or f.motor_overheating or not f.driver_enable_status)
    return ok, m.motor_temp


def j6_recover(p, max_wait=240, log=print):
    log("  J6 fault - widening limit, clearing, cooling")
    p.MotorAngleLimitMaxSpdSet(motor_num=6, max_angle_limit=J6_LIMIT_MAX,
                               min_angle_limit=J6_LIMIT_MIN, max_joint_spd=0x7FFF)
    time.sleep(0.4)
    t0 = time.time()
    while time.time() - t0 < max_wait:
        p.JointConfig(joint_num=6, clear_err=0xAE)
        time.sleep(0.6)
        for _ in range(4):
            p.EnablePiper()
            time.sleep(0.35)
            if all(p.GetArmEnableStatus()):
                break
        ok, temp = j6_health(p)
        if ok and temp <= J6_TEMP_RESUME:
            log(f"  J6 recovered ({temp}C)")
            return True
        log(f"  J6 cooling: {temp}C")
        time.sleep(5)
    return False


# ---------------------------------------------------------------- motion
def _tick_joints(p, j, grip_um, effort=1500):
    p.MotionCtrl_2(0x01, 0x01, SPEED, 0x00)
    p.JointCtrl(*[int(v * 1000) for v in j])
    if grip_um is not None:
        p.GripperCtrl(int(grip_um), effort, 0x01, 0)


def emergency_lift(p, log=print):
    """Raise the shoulder to pull the tool off the table, whatever else is going on."""
    log("  !! table floor breached - emergency lift")
    j = joints(p)
    for step in range(1, 13):
        j2 = j.copy()
        j2[1] = j[1] - step * 2.0          # decreasing J2 raises the arm
        for _ in range(3):
            _tick_joints(p, j2, None)
            time.sleep(DT)
        if ee_z(p) > Z_FLOOR_MM + 20:
            break
    log(f"  lifted to z={ee_z(p):.0f}mm")


def move_joints(p, target, secs, grip_um=None, on_frame=None, guard=True,
                log=print, tol=1.5, settle=0.25):
    """Move to a joint target, streaming it until reached (or secs elapses).

    The FINAL target is streamed unchanged - the Piper runs its own trapezoidal
    trajectory per target, so re-sending an interpolated waypoint every 66 ms
    makes it re-plan constantly and move in visible jerks. Streaming one steady
    target is both smoother and simpler.

    Returns True if it completed, False if the table-floor guard aborted it.
    """
    target = np.asarray(target, dtype=np.float64)
    t0 = time.time()
    reached_at = None
    while time.time() - t0 < secs:
        _tick_joints(p, target, grip_um)
        if guard and ee_z(p) < Z_ABORT_MM:
            emergency_lift(p, log=log)
            return False
        if on_frame:
            on_frame()
        time.sleep(DT)
        if reached_at is None and np.abs(joints(p) - target).max() < tol:
            reached_at = time.time()
        elif reached_at is not None and time.time() - reached_at >= settle:
            break                      # arrived and settled; no need to burn the rest
    return True


def hold_joints(p, target, secs, grip_um=None, on_frame=None, guard=True, log=print):
    n = max(1, int(secs * HZ))
    for _ in range(n):
        _tick_joints(p, target, grip_um)
        if guard and ee_z(p) < Z_ABORT_MM:
            emergency_lift(p, log=log)
            return False
        if on_frame:
            on_frame()
        time.sleep(DT)
    return True


def go_home(p, secs=6.0, log=print):
    """Home to HOME - the one configuration we know is clear of the table.

    Not floor-guarded: homing raises the arm, and the guard's emergency lift
    would fight it if the arm happens to start low.
    """
    log("  homing")
    move_joints(p, HOME, secs, grip_um=None, guard=False, log=log)
    hold_joints(p, HOME, 1.0, grip_um=None, guard=False, log=log)
    j = joints(p)
    off = np.abs(j - np.array(HOME))
    if off.max() > 4.0:
        log(f"  WARNING: home not reached, off by {np.round(off,1)}")
    return j


go_zero = go_home        # backwards-compatible alias


def validate_plan(planner, x, y, z):
    """Reject a target before any motion happens."""
    if z < Z_FLOOR_MM:
        raise TableFloorViolation(f"target z={z:.0f} below floor {Z_FLOOR_MM:.0f}")
    if not planner.reachable(x, y, z):
        raise TableFloorViolation(f"target ({x:.0f},{y:.0f},{z:.0f}) outside sampled workspace")
    return planner.joints_for(x, y, z)
