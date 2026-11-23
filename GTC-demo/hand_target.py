"""Choose where on an open hand to drop the candy (GTC demo; no robot, no network).

The hand service reports the palm centre and the finger base (the centre of
the four knuckles) separately, each with a depth-checked visibility status.
A hand is ready once its finger base is visible and inside the hand
workspace. The candy then goes to the palm centre if that is also visible and
inside the workspace, otherwise to the finger base.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def inside(point: np.ndarray, minimum: Any, maximum: Any) -> bool:
    return bool(np.all(point >= np.asarray(minimum, dtype=np.float64)) and np.all(point <= np.asarray(maximum, dtype=np.float64)))


def choose_drop_target(
    capture: dict[str, Any],
    camera_to_base: Callable[[np.ndarray], np.ndarray],
    workspace_min_mm: Any,
    workspace_max_mm: Any,
) -> dict[str, Any]:
    """Return {"status": "ready", "target", "base_point_mm", ...} or {"status": "waiting", "reason"}.

    camera_to_base converts a camera-frame point (mm) to the robot base frame.
    Only open palms produce a points block, so other gestures keep waiting.
    """
    points = capture.get("points")
    if not points:
        return {"status": "waiting", "reason": "no open palm"}
    finger_base, palm = points.get("finger_base", {}), points.get("palm_center", {})
    if finger_base.get("status") != "visible":
        return {"status": "waiting", "reason": f"finger base not visible ({finger_base.get('status', 'missing')})"}
    finger_base_mm = np.asarray(camera_to_base(np.asarray(finger_base["camera_point_mm"], dtype=np.float64)))
    if not inside(finger_base_mm, workspace_min_mm, workspace_max_mm):
        return {
            "status": "waiting", "reason": "finger base outside the hand workspace",
            "finger_base_mm": finger_base_mm.tolist(),
        }
    result = {"status": "ready", "finger_base_mm": finger_base_mm.tolist()}
    if palm.get("status") == "visible":
        palm_mm = np.asarray(camera_to_base(np.asarray(palm["camera_point_mm"], dtype=np.float64)))
        result["palm_center_mm"] = palm_mm.tolist()
        if inside(palm_mm, workspace_min_mm, workspace_max_mm):
            return {**result, "target": "palm_center", "base_point_mm": palm_mm.tolist()}
        result["palm_note"] = "palm centre outside the hand workspace"
    else:
        result["palm_note"] = f"palm centre not visible ({palm.get('status', 'missing')})"
    return {**result, "target": "finger_base", "base_point_mm": finger_base_mm.tolist()}
