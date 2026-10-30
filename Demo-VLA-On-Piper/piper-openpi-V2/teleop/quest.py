"""Meta Quest 3 WebXR teleop for the Piper: HTTPS page + WS/POST feed + 30 Hz Cartesian control thread.

    python -m teleop.quest --dry-run                 serve the page, print targets, no robot
    python -m teleop.quest [--scale 0.1]             enable, move to READY (joint-space), then drive the arm
    python -m teleop.quest --replay                  synthetic controller trajectory through the same mapping
    python -m teleop.quest --sim                     same, on the MuJoCo twin (sim/): no robot, watch :8092 or the viewer

Right controller - A owns teleop, B owns episodes:
    hold A 3 s = ARM / DISARM (nothing moves until armed)      tap A = discard the running episode
    hold B 3 s = START an episode                              tap B = STOP and SAVE it
    squeeze = clutch (hold to move), trigger = gripper (1.0 = closed), thumbstick click = READY.
Events for the recorder: a_tap / b_hold / b_tap (disarming discards too).
A target that would push a joint past its range (the wrist pitch J5 at +-70 deg is the usual one) is replaced by
the nearest reachable pose every tick, so the arm slides along the limit instead of freezing; the terminal and the
headset show which joint (limit=J5). Rotate the controller back to get the freedom back.
--scale 0.1 moves the robot 10 % of the controller motion (translation and rotation): a safety cap for tests.
The controller moves the wrist centre (TELEOP_PIVOT_MM behind the flange) and rotates the tool about it, so a pure
hand rotation is a wrist-only motion: the base joints stay put.
Leaving XR disarms. The clutch drops and a fresh squeeze is required when the target sits at the lead clamp
for STUCK_S (the firmware IK refusing silently) or the firmware itself reports a collision / e-stop.
Ctrl-C stops streaming, homes slowly and de-torques the arm; so does a crash of the control loop.
"""
import argparse
import asyncio
import json
import os
import queue
import ssl
import threading
import time
import traceback
from collections import deque

import numpy as np
from aiohttp import WSMsgType, web
from scipy.spatial.transform import Rotation as Rot

from piper import config as C
from piper.kinematics import reachable
from piper.robot import Piper, clamp_pose, connect_for_motion, pose_to_xyz_rot, xyz_rot_to_pose

XR_TO_ROBOT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], np.float64)   # x_r=-z_xr, y_r=-x_xr, z_r=y_xr


def ready_pose():
    from piper_sdk.kinematics import C_PiperForwardKinematics
    return np.array(C_PiperForwardKinematics().CalFK(np.radians(C.READY_JOINTS_DEG).tolist())[-1])


def arm_status_name(s) -> str:
    """piper_sdk prints its enum as 'NO_SOLUTION(0x2)' (tests: 'ArmStatus.NO_SOLUTION') -> 'NO_SOLUTION'."""
    return str(s).split("(")[0].rsplit(".", 1)[-1]


def _plain(v):
    """numpy scalars/arrays -> JSON-serializable Python. See snapshot() for why this is not optional."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


class QuestTeleop:
    def __init__(self, robot: Piper | None = None, scale=C.TELEOP_SCALE, armed=False, rot_scale=None):
        self.robot = robot
        self.scale = scale
        self.rot_scale = scale if rot_scale is None else rot_scale
        self.events = queue.Queue()
        self.armed = armed
        self.arm_progress = 0.0
        self.engaged = False
        self.grip_cmd_mm = C.GRIP_OPEN_MM
        self.target_pose = None
        self.status = "DISARMED: hold A 3 s"
        self.fault = None            # set when the control loop died on an exception; main() then rests the arm
        self.packets = 0
        self.transport = "none"
        self.tracked = False
        self.inputs = {"squeeze": 0.0, "trigger": 0.0, "a": False}
        self.telemetry = {}
        self.err_mm, self.err_deg = 0.0, 0.0
        self.lead_limited = False
        self.limit = []              # joints the target would push past their range (e.g. ['J5']); the target is
                                     # projected onto the nearest reachable pose instead of being refused
        self.stuck = False
        self.no_feedback = False
        self.j6_margin = None
        self.arm_status = None       # firmware verdict (ArmStatus name), polled at 1 Hz with the telemetry
        self.disabled, self.faults = [], []   # joints the driver has cut / firmware faults, polled at 1 Hz
        self.rec_state = {}
        self._stuck_since = None
        self._need_resqueeze = False
        self._clock = time.monotonic     # tests drive _tick() with a fake clock
        self._lock = threading.Lock()
        self._packet, self._packet_t = None, 0.0
        self._pkt_times = deque(maxlen=200)
        self._basis = XR_TO_ROBOT
        self._rezero = True
        self._ref = None
        self._buttons = {"a": False, "b": False, "thumb": False}
        self._a_since, self._a_consumed = None, False
        self._b_since, self._b_consumed, self.b_progress = None, False, 0.0
        self._go_ready = False
        self._disarm = False
        self._paused = False
        self._telemetry_t = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # ---- lifecycle -----------------------------------------------------------
    def start(self):
        if self.robot is not None:
            self.grip_cmd_mm = self.robot.grip_mm()      # keep the opening the gripper has, never snap it open
        self._goto_ready()
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.status = "stopped"

    def snapshot(self) -> dict:
        """Both the headset feed and the recorder's /status serialize this, so ONE numpy scalar in here
        500s every request from then on: the arm stops responding and no episode can be started, with
        nothing on the headset to say why. numpy 2 calls np.bool_ 'bool', so the traceback reads
        "Object of type bool is not JSON serializable" - which reads like nonsense. Hence _plain()."""
        now = self._clock()
        with self._lock:
            age = now - self._packet_t if self._packet is not None else None
        rate = sum(1 for t in self._pkt_times if now - t < 1.0)
        return _plain({"armed": self.armed, "engaged": self.engaged, "arm_progress": round(self.arm_progress, 2),
                "b_progress": round(self.b_progress, 2),
                "status": self.status, "pkts": self.packets, "transport": self.transport,
                "rate_hz": rate, "pkt_age_s": None if age is None else round(age, 2), "tracked": self.tracked,
                "squeeze": round(self.inputs["squeeze"], 2), "trigger": round(self.inputs["trigger"], 2),
                "a": self.inputs["a"], "grip_cmd_mm": round(float(self.grip_cmd_mm), 1), "scale": self.scale,
                "target_pose": None if self.target_pose is None else np.round(self.target_pose, 1).tolist(),
                "err_mm": round(self.err_mm, 1), "err_deg": round(self.err_deg, 1),
                "lead_limited": self.lead_limited, "limit": self.limit, "stuck": self.stuck, "no_feedback": self.no_feedback,
                "j6_margin_deg": self.j6_margin, "arm_status": self.arm_status,
                "disabled": self.disabled, "faults": self.faults,
                "recording": self.rec_state.get("recording"), "episode": self.rec_state.get("episode"),
                "frames": self.rec_state.get("frames"), **self.telemetry})

    # ---- input (any thread) --------------------------------------------------
    def feed(self, msg: dict, transport="ws") -> dict:
        self.transport = transport
        ev = msg.get("event")
        if ev == "session_start":
            self._rezero = True
            self.events.put(f"session_start:{msg.get('mode')}")
        elif ev == "session_end":
            self._disarm = True          # operator left XR: nothing may move on a stray A-hold later
            self.events.put("session_end")
        elif ev == "hello":
            self.events.put(f"hello via {transport}")
        else:
            now = self._clock()
            with self._lock:
                self._packet, self._packet_t = msg, now
            self._pkt_times.append(now)
            self.packets += 1
        return self.snapshot()

    def _parse(self, msg):
        head, right = msg.get("head"), msg.get("right")
        if self._rezero and head and head.get("q"):
            fwd = XR_TO_ROBOT @ Rot.from_quat(head["q"]).apply([0.0, 0.0, -1.0])
            yaw = np.degrees(np.arctan2(fwd[1], fwd[0]))
            self._basis = Rot.from_euler("z", -yaw, degrees=True).as_matrix() @ XR_TO_ROBOT
            self._rezero = False
            self.events.put(f"yaw_zeroed:{yaw:.0f}")
        if not right:
            return None
        self.inputs = {"squeeze": float(right.get("squeeze", 0.0)), "trigger": float(right.get("trigger", 0.0)),
                       "a": bool(right.get("a"))}
        self._buttons_update(right)
        if not right.get("p"):
            return None
        pos = self._basis @ np.asarray(right["p"], np.float64) * 1000.0
        rot = Rot.from_matrix(self._basis @ Rot.from_quat(right["q"]).as_matrix() @ self._basis.T)
        if not np.isfinite(pos).all():
            raise ValueError(f"non-finite position {right['p']}")
        return pos, rot

    def _buttons_update(self, right):
        now = self._clock()
        a = bool(right.get("a"))
        if a and self._a_since is None:
            self._a_since, self._a_consumed = now, False
        if a:
            held = now - self._a_since
            self.arm_progress = min(1.0, held / C.TELEOP_ARM_HOLD_S)
            if held >= C.TELEOP_ARM_HOLD_S and not self._a_consumed:
                self._a_consumed = True
                self._set_armed(not self.armed)
        elif self._a_since is not None:
            if not self._a_consumed and self.armed:          # released before the arm/disarm threshold: a tap
                self.events.put("a_tap")                     # (the recorder discards the running episode)
            self._a_since, self.arm_progress = None, 0.0
        b = bool(right.get("b"))                             # B owns episodes: hold = start, tap = stop and save
        if b and self._b_since is None:
            self._b_since, self._b_consumed = now, False
        if b:
            held = now - self._b_since
            self.b_progress = min(1.0, held / C.TELEOP_EPISODE_HOLD_S) if self.armed else 0.0
            if held >= C.TELEOP_EPISODE_HOLD_S and not self._b_consumed and self.armed:
                self._b_consumed = True
                self.events.put("b_hold")
        elif self._b_since is not None:
            if not self._b_consumed and self.armed:
                self.events.put("b_tap")
            self._b_since, self.b_progress = None, 0.0
        pressed = bool(right.get("thumb"))
        if pressed and not self._buttons["thumb"] and self.armed:
            self.events.put("thumb")
            self._go_ready = True
        self._buttons["thumb"] = pressed

    def _set_armed(self, armed):
        self.armed = armed
        self._release("ARMED: squeeze to move" if armed else "DISARMED: hold A 3 s")
        self.events.put("armed" if armed else "disarmed")

    # ---- 30 Hz control loop --------------------------------------------------
    def _loop(self):
        dt, next_t = 1.0 / C.CONTROL_HZ, time.monotonic()
        try:
            while not self._stop.is_set():
                self._tick()
                next_t = max(next_t + dt, time.monotonic())      # no catch-up burst after a blocking READY move
                time.sleep(max(0.0, next_t - time.monotonic()))
        except Exception as e:
            # a half-dead controller must not keep the arm: stop streaming, say so, main() rests the arm
            self.fault = f"{type(e).__name__}: {e}"
            self.engaged, self._ref, self.status = False, None, f"CONTROL FAULT: {self.fault}"
            self.events.put(f"fault:{self.fault}")
            traceback.print_exc()

    def _tick(self):
        now = self._clock()
        if self._go_ready:
            self._ready()
        if self._disarm:
            self._disarm = False
            if self.armed:
                self._set_armed(False)
        meas_xyz = meas_rot = None
        if self.robot is not None:
            pose = self.robot.pose()
            self.no_feedback = not np.any(np.abs(pose) > 1e-6)
            if not self.no_feedback:
                meas_xyz, meas_rot = pose_to_xyz_rot(pose)
            if now - self._telemetry_t > 1.0:
                self._telemetry_t = now
                self._poll_arm(pose)
        with self._lock:
            msg, t_pkt = self._packet, self._packet_t
        fresh = msg is not None and now - t_pkt < C.TELEOP_WATCHDOG_S
        ctrl = None
        if fresh:
            try:
                ctrl = self._parse(msg)
            except Exception as e:                       # malformed packet: drop it and hold, keep running
                with self._lock:
                    self._packet = None
                self.events.put(f"bad packet dropped: {e!r}")
        else:
            self._a_since, self.arm_progress = None, 0.0  # an A-hold interrupted by a packet gap does not count
        self.tracked = ctrl is not None
        if ctrl is None:
            self._release("hold: no controller packets" if not fresh else "hold: controller not tracked")
        elif not self.armed:
            self._release("DISARMED: hold A 3 s")
        elif self.no_feedback:
            self._release("NO ARM FEEDBACK (arm powered? CAN up?)")
        else:
            pos, rot = ctrl
            squeeze = self.inputs["squeeze"] > 0.5
            if not squeeze:
                self._need_resqueeze = False
                if self.engaged and self.lead_limited and meas_xyz is not None:   # the IK was refusing: do not
                    self._cmd_xyz, self._cmd_rot = meas_xyz.copy(), meas_rot      # leave a target 30 mm ahead
                self._release("ARMED: squeeze to move")
            elif not self.engaged and not self._paused and not self._need_resqueeze:
                self._ref = (pos, rot, self._cmd_xyz.copy(), self._cmd_rot)
                self.engaged, self.status, self.stuck = True, "ENGAGED", False
                self.events.put("engaged")
            if self.engaged:
                rp, rr, bp, br = self._ref
                tgt_rot = Rot.from_rotvec(self.rot_scale * (rot * rr.inv()).as_rotvec()) * br
                pivot = np.array([0.0, 0.0, -C.TELEOP_PIVOT_MM])          # wrist centre in the flange frame: the
                tgt_xyz = (bp + br.apply(pivot) + self.scale * (pos - rp)   # hand moves it, hand rotations turn the
                           - tgt_rot.apply(pivot))                         # tool about it (wrist-only motion)
                self._cmd_xyz, self._cmd_rot, self.lead_limited = clamp_pose(
                    self._cmd_xyz, self._cmd_rot, tgt_xyz, tgt_rot, meas_xyz, meas_rot)
                if meas_xyz is not None:             # a joint would leave its range: send the nearest pose the arm
                    self._cmd_xyz, self._cmd_rot, self.limit = reachable(   # can take, so it slides along the
                        self._cmd_xyz, self._cmd_rot, self.robot.joints_deg())   # limit instead of freezing
                    if self.limit:                   # at a limit the wish is legitimately far ahead: judge the
                        rv = (meas_rot.inv() * self._cmd_rot).as_rotvec(degrees=True)   # stuck timer on what we SEND
                        self.lead_limited = bool(np.linalg.norm(self._cmd_xyz - meas_xyz) > C.LEAD_MAX_MM
                                                 or np.linalg.norm(rv) > C.LEAD_MAX_DEG)   # np.bool_ otherwise
            trig = self.inputs["trigger"]
            trig = 0.0 if trig < C.TRIGGER_DEADZONE else min(trig, 1.0)
            self.grip_cmd_mm = C.GRIP_OPEN_MM - trig * (C.GRIP_OPEN_MM - C.GRIP_CLOSED_MM)
        # the firmware IK refuses silently: if the target sits at the lead clamp too long, snap it back
        # to the real pose, drop the clutch and require a fresh squeeze (self-healing, no lurch)
        if self.engaged and self.lead_limited:
            self._stuck_since = self._stuck_since or now
            if now - self._stuck_since > C.STUCK_S:
                self._declutch("IK REFUSED - release and re-squeeze", meas_xyz, meas_rot)
        else:
            self._stuck_since = None
        # the firmware's own verdict: while it reports a collision / e-stop the target tracks the real pose
        if self.arm_status in C.FAULT_STATUS:
            self._declutch(f"{self.arm_status} - release and re-squeeze", meas_xyz, meas_rot)
        if self.disabled:                                # a driver cut a joint: nothing can follow, say so
            self._declutch(f"{'+'.join(self.disabled)} NOT ENABLED - Ctrl-C, then tools/rest.py", meas_xyz, meas_rot)
        if not self.engaged:
            self.lead_limited, self.limit = False, []
        if meas_xyz is not None:
            self.err_mm = float(np.linalg.norm(self._cmd_xyz - meas_xyz))
            self.err_deg = float(np.linalg.norm((meas_rot.inv() * self._cmd_rot).as_rotvec(degrees=True)))
        self.target_pose = xyz_rot_to_pose(self._cmd_xyz, self._cmd_rot)
        if self.robot is not None and not self.no_feedback:
            self.robot.send_pose(self._cmd_xyz, self.target_pose[3:], self.grip_cmd_mm)

    def _release(self, status):
        if self.engaged:
            self.engaged, self._ref = False, None
            self.events.put("released")
        if not self._paused:
            self.status = status

    def _declutch(self, status, meas_xyz, meas_rot):
        """Guard tripped: target back to the real pose, clutch off, fresh squeeze required; the headset
        shows STUCK and the status says why."""
        if meas_xyz is not None:
            self._cmd_xyz, self._cmd_rot = meas_xyz.copy(), meas_rot
        self._need_resqueeze = True
        if not self.stuck:
            self.stuck = True
            self.events.put("stuck")
        self._release(status)

    def _goto_ready(self):
        """Joint-space move to READY (the firmware IK refuses Cartesian targets from the folded HOME), then
        restart the Cartesian target from the pose the arm reports there."""
        if self.robot is not None:
            print(f"moving to READY {C.READY_JOINTS_DEG} (joint-space, {C.SPEED_HOME} %)")
            ok = self.robot.ready(self.grip_cmd_mm)
            pose = self.robot.pose()
            print(f"at READY: joints {np.round(self.robot.joints_deg(), 1).tolist()}  pose {np.round(pose, 1).tolist()}")
            if not ok:                                   # never stream from a pose the arm could not reach
                raise SystemExit(f"ABORT: READY not reached - enabled {self.robot.enabled()} faults "
                                 f"{self.robot.faults() or 'none'}. A joint on a hard stop / cut by its driver? "
                                 "(J6 zero: tools/zero_j6.py). Resting.")
        else:
            pose = ready_pose()
        self._cmd_xyz, self._cmd_rot = pose_to_xyz_rot(pose)
        self.target_pose = xyz_rot_to_pose(self._cmd_xyz, self._cmd_rot)

    def _ready(self):
        """Thumbstick click: back to READY (HOME is folded and cannot be driven in Cartesian mode)."""
        self._go_ready = False
        self._release("moving to READY")
        self._goto_ready()
        self.events.put("ready")

    def _poll_arm(self, pose):
        """1 Hz: J6 temperature pause / resume, wrist-roll margin, firmware verdict, telemetry for the ack."""
        self._check_temp()
        j = self.robot.joints_deg()
        lo, hi = C.JOINT_LIMITS_DEG[5]
        self.j6_margin = round(min(j[5] - lo, hi - j[5]), 1)
        self.arm_status = arm_status_name(self.robot.arm_status())
        disabled = [f"J{i + 1}" for i, e in enumerate(self.robot.enabled()[:6]) if not e]
        faults = list(self.robot.faults())
        if disabled != self.disabled or faults != self.faults:
            self.events.put(f"arm: disabled {disabled or 'none'} faults {faults or 'none'}")
        self.disabled, self.faults = disabled, faults
        self.telemetry = {"pose": np.round(pose, 1).tolist(), "joints": np.round(j, 1).tolist(),
                          "grip_mm": round(self.robot.grip_mm(), 1), "j6_c": self.robot.j6_temp()}

    def _check_temp(self):
        temp = self.robot.j6_temp()
        if temp >= C.J6_TEMP_PAUSE_C and not self._paused:
            self._paused = True
            self._release(f"PAUSED: J6 {temp} C")
            self.status = f"PAUSED: J6 {temp} C"
            self.events.put("j6_pause")
        elif self._paused and temp <= C.J6_TEMP_RESUME_C:
            self._paused = False
            self.status = "hold: J6 cooled"
            self.events.put("j6_resume")


# ---- HTTPS server: page, WS feed, POST feed (fallback when the Quest browser refuses wss) ----
def packet(raw):
    """Decoded JSON object, or None: a malformed packet must never take the server or the control loop down."""
    try:
        msg = json.loads(raw)
    except ValueError:
        return None
    return msg if isinstance(msg, dict) else None


def make_app(teleop: QuestTeleop) -> web.Application:
    bad = {"error": "packet must be a JSON object"}

    async def index(_):
        return web.FileResponse(os.path.join(C.STATIC_DIR, "index.html"))

    async def status(_):
        return web.json_response(teleop.snapshot())

    async def pose(request):
        msg = packet(await request.read())
        if msg is None:
            teleop.events.put(f"bad packet from {request.remote} (400)")
            return web.json_response(bad, status=400)
        return web.json_response(teleop.feed(msg, "http"))

    async def ws(request):
        sock = web.WebSocketResponse(heartbeat=2.0)
        await sock.prepare(request)
        teleop.events.put(f"ws_connect from {request.remote}")
        async for m in sock:
            if m.type == WSMsgType.TEXT:
                msg = packet(m.data)
                if msg is None:
                    teleop.events.put(f"bad ws packet from {request.remote}")
                await sock.send_json(bad if msg is None else teleop.feed(msg, "ws"))
        teleop.events.put("ws_disconnect")
        return sock

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/status", status), web.post("/pose", pose), web.get("/ws", ws)])
    return app


def serve(teleop: QuestTeleop, port=C.TELEOP_PORT):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(C.TELEOP_CERT, C.TELEOP_KEY)

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(make_app(teleop))
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, "0.0.0.0", port, ssl_context=ctx).start())
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    print(f"open {C.TELEOP_URL} in the Quest browser (accept the cert), tap ENTER, hold A 3 s to arm")


def status_line(teleop: QuestTeleop) -> str:
    s = teleop.snapshot()
    net = f"{s['transport']} {s['rate_hz']:2d}/s age={s['pkt_age_s']}" if s["pkts"] else "NO PACKETS YET"
    arm = ("ENGAGED" if s["engaged"] else "ARMED  ") if s["armed"] else f"DISARMED {s['arm_progress']:.1f}"
    fw = "" if s["arm_status"] in (None, "NORMAL") else f" fw={s['arm_status']}"
    fw += f" limit={'+'.join(s['limit'])}" if s.get("limit") else ""
    fw += f" DISABLED={'+'.join(s['disabled'])}" if s.get("disabled") else ""
    fw += f" faults={'+'.join(s['faults'])}" if s.get("faults") else ""
    line = (f"[{net}] ctrl={'tracked' if s['tracked'] else 'none   '} sq={s['squeeze']:.2f} tr={s['trigger']:.2f} "
            f"A={int(s['a'])} | {arm} | {s['status']:24s}{fw} | grip={s['grip_cmd_mm']:4.0f} tgt={s['target_pose']}")
    if "pose" in s:
        line += f"\n      meas={s['pose']} joints={s['joints']} grip={s['grip_mm']} J6={s['j6_c']}C"
    return line


# ---- synthetic controller (self-test, --fake recording) -----------------------------
def synthetic_feed(teleop: QuestTeleop, secs, stop: threading.Event, head_yaw_deg=30.0,
                   amp_m=0.05, period_s=4.0, yaw_amp_deg=10.0, hz=60):
    """Circle of radius amp_m in front of a head yawed head_yaw_deg (tests the yaw re-zero),
    controller yaw wobble +-yaw_amp_deg, squeeze held from 1 s to secs-1, trigger sweeping."""
    head_rot = Rot.from_euler("y", head_yaw_deg, degrees=True)
    head = {"p": [0.0, 1.6, 0.0], "q": head_rot.as_quat().tolist()}
    teleop.feed({"event": "session_start"}, "fake")
    t0 = time.monotonic()
    while not stop.is_set() and (t := time.monotonic() - t0) < secs:
        w = 2 * np.pi * (t - 1.0) / period_s if t > 1.0 else 0.0
        p_local = np.array([amp_m * np.sin(w), 0.0, -amp_m * (1 - np.cos(w))])
        q_local = Rot.from_euler("y", yaw_amp_deg * np.sin(w), degrees=True)
        right = {"p": (head_rot.apply(p_local) + [0.2, 1.2, -0.4]).tolist(),
                 "q": (head_rot * q_local).as_quat().tolist(),
                 "trigger": 0.5 + 0.5 * np.sin(w / 2), "squeeze": 1.0 if 1.0 < t < secs - 1.0 else 0.0,
                 "a": False, "b": False, "thumb": False, "axes": [0.0, 0.0]}
        teleop.feed({"t": t, "head": head, "right": right}, "fake")
        time.sleep(1.0 / hz)


def replay(secs=9.0, scale=1.0):
    teleop = QuestTeleop(None, scale=scale, armed=True).start()
    stop = threading.Event()
    threading.Thread(target=synthetic_feed, args=(teleop, secs, stop), daemon=True).start()
    pivot = np.array([0.0, 0.0, -C.TELEOP_PIVOT_MM])

    def wrist_centre(tp):                                     # the point the controller moves
        xyz, r = pose_to_xyz_rot(tp)
        return xyz + r.apply(pivot)
    p0 = teleop.target_pose.copy()
    rows, t0 = [], time.monotonic()
    while time.monotonic() - t0 < secs:
        time.sleep(0.2)
        tp = teleop.target_pose
        d = np.concatenate([wrist_centre(tp) - wrist_centre(p0), tp[3:] - p0[3:]])
        d[3:] = (d[3:] + 180.0) % 360.0 - 180.0               # Euler deltas wrap at +-180 (where the tool sits at READY)
        rows.append(d)
        print(f"t={time.monotonic() - t0:4.1f}s {teleop.status:26s} dxyz={np.round(d[:3], 1)} "
              f"drpy={np.round(d[3:], 1)} grip={teleop.grip_cmd_mm:5.1f}")
    stop.set()
    teleop.stop()
    d = np.array(rows)
    exp_x, exp_y, exp_yaw = 2000 * 0.05 * scale, -1000 * 0.05 * scale, 10.0 * scale
    got = (d[:, 0].max(), d[:, 1].min(), d[:, 5].max())
    ok = abs(got[0] - exp_x) < 5 and abs(got[1] - exp_y) < 5 and abs(got[2] - exp_yaw) < 1.5 and abs(d[:, 2]).max() < 2
    print(f"\nforward 2A -> dx max {got[0]:.1f} (expect {exp_x:.0f});  right A -> dy min {got[1]:.1f} "
          f"(expect {exp_y:.0f});  yaw max {got[2]:.1f} (expect {exp_yaw:.0f});  |dz| max {abs(d[:, 2]).max():.1f} (expect 0)")
    print("REPLAY SELF-TEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="serve + print targets, no robot")
    ap.add_argument("--replay", action="store_true", help="synthetic controller self-test, no robot, no server")
    ap.add_argument("--scale", type=float, default=C.TELEOP_SCALE, help="robot motion per controller motion, e.g. 0.1")
    ap.add_argument("--rot-scale", type=float, default=None, help="rotation scale (default = --scale; 0 freezes rotation)")
    ap.add_argument("--port", type=int, default=C.TELEOP_PORT)
    ap.add_argument("--sim", action="store_true", help="MuJoCo twin instead of the CAN bus (sim/)")
    args = ap.parse_args()
    if args.replay:
        raise SystemExit(0 if replay(scale=args.scale) else 1)
    if args.sim:
        from sim.piper_sim import connect_sim
    robot = None if args.dry_run else (connect_sim() if args.sim else connect_for_motion())
    teleop = QuestTeleop(robot, scale=args.scale, rot_scale=args.rot_scale).start()
    serve(teleop, args.port)
    print(f"scale={args.scale} rot_scale={teleop.rot_scale}  (robot moves {args.scale * 100:.0f} % / rotates "
          f"{teleop.rot_scale * 100:.0f} % of the controller)")
    try:
        while teleop.fault is None:
            time.sleep(1.0)
            while not teleop.events.empty():
                print("event:", teleop.events.get())
            print(status_line(teleop))
        print("\ncontrol loop fault: stopping stream, homing, de-torquing")
    except KeyboardInterrupt:
        print("\nCtrl-C: stopping stream, homing, de-torquing")
    finally:
        teleop.stop()
        if robot is not None:
            robot.rest()


if __name__ == "__main__":
    main()
