"""AgileX Piper (REAL hardware) — Asimov MCP toolset.

Maps the Asimov SDK onto the physical AgileX Piper 6-DOF arm over CAN
(piper_sdk), following the same ToolsetMCPServer pattern as the sim toolset and
the reference Asimov samples. A single PIPER_MODE switch selects the backend:

    PIPER_MODE=mock  (default) — canned state, NO hardware / CAN bus.
    PIPER_MODE=real            — the physical Piper over CAN (piper_sdk).

Safety (mirrors the Asimov FR3 real toolset):
  * Reads are always allowed. MOTION is OBSERVE-ONLY by default and requires
    PIPER_ALLOW_MOTION=1. Without it, enable/move/gripper tools refuse.
  * emergency_stop() is always available.
  * In real mode the arm must first be put in Slave mode and enabled (see the
    toolset README / SFF-AgileX-Piper prepare-jetson/SETUP.md).

Tool-surface units: joints in DEGREES, Cartesian x/y/z in METERS, orientation
rx/ry/rz in DEGREES, gripper opening as a 0..1 fraction.
"""
import asyncio
import json
import os

from asimov.robot_agent_toolset import ToolsetMCPServer, tool_error, tool_success

from backends import MockPiperBackend, RealPiperBackend

MODE = os.getenv("PIPER_MODE", "mock").strip().lower()
ALLOW_MOTION = os.getenv("PIPER_ALLOW_MOTION", "0").strip() in ("1", "true", "yes")
CAN_IFACE = os.getenv("PIPER_CAN_IFACE", "can0")
SPEED = int(os.getenv("PIPER_SPEED_PERCENT", "30"))
CAMERA_ON = os.getenv("PIPER_CAMERA", "on").strip().lower() in ("1", "true", "yes", "on")
API_HOST = os.getenv("PIPER_TOOLSET_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PIPER_TOOLSET_API_PORT", "8212"))

INSTRUCTIONS = f"""AgileX Piper 6-DOF robotic arm — REAL hardware toolset (mode={MODE}).

Coordinate frame: x forward, y left, z up, in METERS from the arm base. Joint
angles in DEGREES (joint1 base .. joint6 wrist). Gripper opening 0..1.

Motion is {'ENABLED' if ALLOW_MOTION else 'OBSERVE-ONLY (disabled)'}. Always call
get_status() first. In real mode call enable_arm() once before any motion. The
arm does its own inverse kinematics for move_to_pose. emergency_stop() halts the
arm immediately. Motions are long-running — results come back via poll_status.
"""


def _make_backend():
    if MODE == "real":
        return RealPiperBackend(can_iface=CAN_IFACE, speed_percent=SPEED)
    return MockPiperBackend()


class PiperRealToolset(ToolsetMCPServer):
    def __init__(self):
        super().__init__(
            "piper_real",
            f"AgileX Piper real-hardware toolset ({MODE} mode)",
            version="1.0.0",
            instructions=INSTRUCTIONS,
        )
        self.backend = _make_backend()
        info = self.backend.connect()
        self._conn_info = info

        # Optional RealSense D405 wrist camera (lazy; degrades gracefully).
        self.camera = None
        self._camera_error = None
        if CAMERA_ON:
            try:
                from camera import PiperCamera
                self.camera = PiperCamera()
                cam_info = self.camera.start()
                print(f"[piper-real] camera: {cam_info.get('name')} serial={cam_info.get('serial')}")
            except Exception as e:  # noqa: BLE001
                self.camera = None
                self._camera_error = str(e)
                print(f"[piper-real] camera unavailable: {e}")

        self._register_tools()
        self._register_state_endpoint()
        self._register_camera_endpoint()

    def _deny_if_locked(self, action):
        if not ALLOW_MOTION:
            return tool_error(action, "Motion is OBSERVE-ONLY. Set PIPER_ALLOW_MOTION=1 "
                                      "(and ensure the arm is in Slave mode + enabled) to move it.")
        return None

    def _register_state_endpoint(self):
        b = self.backend

        @self.app.get("/state")
        async def get_state():
            return b.get_status()

    def _register_camera_endpoint(self):
        @self.app.get("/camera/snapshot")
        async def camera_snapshot():
            from fastapi.responses import FileResponse, JSONResponse
            if self.camera is None:
                return JSONResponse({"error": self._camera_error or "camera off"}, status_code=503)
            meta = self.camera.capture("http")
            return FileResponse(meta["path"], media_type="image/jpeg")

    def _register_tools(self):
        b = self.backend

        # ---------------- sync reads (always allowed) --------------------
        @self.sync_tool("Get overall arm status: mode, enabled, joint angles, CAN fps and telemetry rate.")
        async def get_status() -> dict:
            return tool_success("get_status", json.dumps(b.get_status()))

        @self.sync_tool("Get the 6 joint angles in degrees (joint1..joint6).")
        async def get_joint_positions() -> dict:
            j = {f"joint{i+1}": round(v, 1) for i, v in enumerate(b.get_joint_positions_deg())}
            return tool_success("get_joint_positions", json.dumps(j))

        @self.sync_tool("Get the end-effector pose: x, y, z in meters and rx, ry, rz in degrees.")
        async def get_ee_pose() -> dict:
            return tool_success("get_ee_pose", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                           for k, v in b.get_ee_pose().items()}))

        @self.sync_tool("Get the gripper opening fraction (1.0 = fully open, 0.0 = closed).")
        async def get_gripper() -> dict:
            return tool_success("get_gripper", json.dumps({"opening": round(b.get_gripper_opening(), 2)}))

        @self.sync_tool("Get CAN connection info (fps, telemetry Hz) and any connection warnings.")
        async def get_connection() -> dict:
            return tool_success("get_connection", json.dumps(self._conn_info))

        # ---------------- camera (RealSense D405 wrist cam) --------------
        @self.sync_tool(
            "Capture an image from the wrist camera. Saves a color snapshot and "
            "returns its file path plus the depth (mm) at the image centre. "
            "Served live at GET /camera/snapshot."
        )
        async def capture_image() -> dict:
            if self.camera is None:
                return tool_error("capture_image", f"Camera unavailable: {self._camera_error or 'disabled'}")
            try:
                meta = self.camera.capture("tool")
                return tool_success("capture_image", json.dumps(meta))
            except Exception as e:  # noqa: BLE001
                return tool_error("capture_image", str(e))

        @self.sync_tool(
            "Read the camera depth (millimeters) at a normalized image point "
            "(u,v in 0..1; 0.5,0.5 = centre). Useful to gauge how far an object is."
        )
        async def get_scene_depth(u: float = 0.5, v: float = 0.5) -> dict:
            if self.camera is None:
                return tool_error("get_scene_depth", f"Camera unavailable: {self._camera_error or 'disabled'}")
            try:
                return tool_success("get_scene_depth", json.dumps(self.camera.depth_at(u, v)))
            except Exception as e:  # noqa: BLE001
                return tool_error("get_scene_depth", str(e))

        # ---------------- emergency stop (always allowed) ----------------
        @self.sync_tool("EMERGENCY STOP: immediately halt the arm and disable motion. Use for 'stop'/'halt'.")
        async def emergency_stop() -> dict:
            return tool_success("emergency_stop", json.dumps(b.emergency_stop()))

        # ---------------- gated motion -----------------------------------
        @self.async_tool("Enable the arm for CAN motion control (required once before moving in real mode).")
        async def enable_arm() -> dict:
            denied = self._deny_if_locked("enable_arm")
            if denied:
                return denied
            return tool_success("enable_arm", json.dumps(b.enable()))

        @self.async_tool("Move the arm to its zero (folded) home position. Long-running.")
        async def home() -> dict:
            denied = self._deny_if_locked("home")
            if denied:
                return denied
            res = b.move_joints_deg([0, 0, 0, 0, 0, 0])
            return tool_success("home", json.dumps(res)) if res.get("ok") else tool_error("home", res.get("detail", "failed"))

        @self.async_tool(
            "Set/move the arm to the ZERO position: all six joints to 0 degrees "
            "(the arm's standard reference pose, per AgileX's draw_square.py). "
            "WARNING: this is a full-range move from wherever the arm currently is "
            "— make sure the workspace is clear. Long-running."
        )
        async def go_to_zero() -> dict:
            denied = self._deny_if_locked("go_to_zero")
            if denied:
                return denied
            res = b.go_to_zero()
            return tool_success("go_to_zero", json.dumps(res)) if res.get("ok") else tool_error("go_to_zero", res.get("detail", "failed"))

        @self.async_tool("Move the 6 joints to absolute angles in DEGREES (joint1..joint6). Long-running.")
        async def move_to_joint_positions(joint1: float, joint2: float, joint3: float,
                                          joint4: float, joint5: float, joint6: float) -> dict:
            denied = self._deny_if_locked("move_to_joint_positions")
            if denied:
                return denied
            res = b.move_joints_deg([joint1, joint2, joint3, joint4, joint5, joint6])
            return tool_success("move_to_joint_positions", json.dumps(res)) if res.get("ok") else tool_error("move_to_joint_positions", res.get("detail", "failed"))

        @self.async_tool(
            "Move the gripper to an absolute Cartesian pose (arm solves IK). x,y,z in meters; "
            "optional rx,ry,rz orientation in degrees (default top-down grasp). Long-running."
        )
        async def move_to_pose(x: float, y: float, z: float,
                               rx: float = 0.0, ry: float = 85.0, rz: float = 0.0) -> dict:
            denied = self._deny_if_locked("move_to_pose")
            if denied:
                return denied
            res = b.move_pose(x, y, z, rx, ry, rz, linear=True)
            return tool_success("move_to_pose", json.dumps(res)) if res.get("ok") else tool_error("move_to_pose", res.get("detail", "failed"))

        @self.async_tool("Move the gripper by a relative offset (dx, dy, dz) in meters. Long-running.")
        async def move_relative(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> dict:
            denied = self._deny_if_locked("move_relative")
            if denied:
                return denied
            p = b.get_ee_pose()
            if not p.get("telemetry_hz", 1):
                return tool_error("move_relative", "No end-pose telemetry (Hz=0) — cannot compute relative move. Check Slave mode.")
            res = b.move_pose(p["x"] + dx, p["y"] + dy, p["z"] + dz, p["rx"], p["ry"], p["rz"], linear=True)
            return tool_success("move_relative", json.dumps(res)) if res.get("ok") else tool_error("move_relative", res.get("detail", "failed"))

        @self.async_tool("Open the gripper fully.")
        async def open_gripper() -> dict:
            denied = self._deny_if_locked("open_gripper")
            if denied:
                return denied
            return tool_success("open_gripper", json.dumps(b.set_gripper(1.0)))

        @self.async_tool("Close the gripper (grasp).")
        async def close_gripper() -> dict:
            denied = self._deny_if_locked("close_gripper")
            if denied:
                return denied
            return tool_success("close_gripper", json.dumps(b.set_gripper(0.0)))

        @self.async_tool(
            "Set the gripper opening to a fraction 0..1 (1 open, 0 closed)."
        )
        async def set_gripper(opening: float) -> dict:
            denied = self._deny_if_locked("set_gripper")
            if denied:
                return denied
            return tool_success("set_gripper", json.dumps(b.set_gripper(opening)))


if __name__ == "__main__":
    server = PiperRealToolset()
    print(f"[piper-real] MCP on {API_HOST}:{API_PORT}  mode={MODE}  motion={'ENABLED' if ALLOW_MOTION else 'OBSERVE-ONLY'}")
    server.run(host=API_HOST, port=API_PORT)
