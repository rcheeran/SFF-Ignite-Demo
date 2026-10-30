"""Pure Hand Landmarker result serialization and drawing helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
PALM_KNUCKLE_LANDMARKS = (5, 9, 13, 17)
FINGER_LANDMARKS = (
    (5, 6, 8),
    (9, 10, 12),
    (13, 14, 16),
    (17, 18, 20),
)
MINIMUM_FINGER_ANGLE_DEG = 150.0
MINIMUM_TIP_EXTENSION_RATIO = 1.1


def _category_value(category: Any, name: str, fallback: Any) -> Any:
    value = getattr(category, name, fallback)
    return fallback if value is None else value


def extended_finger_count(landmarks: list[dict[str, float]]) -> int:
    points = np.asarray(
        [[point["x"], point["y"], point["z"]] for point in landmarks],
        dtype=np.float64,
    )
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        return 0
    wrist = points[0]
    count = 0
    for mcp_index, pip_index, tip_index in FINGER_LANDMARKS:
        mcp = points[mcp_index]
        pip = points[pip_index]
        tip = points[tip_index]
        first = mcp - pip
        second = tip - pip
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= 1e-9:
            continue
        cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosine)))
        pip_radius = float(np.linalg.norm(pip - wrist))
        tip_radius = float(np.linalg.norm(tip - wrist))
        if (
            angle >= MINIMUM_FINGER_ANGLE_DEG
            and tip_radius >= pip_radius * MINIMUM_TIP_EXTENSION_RATIO
        ):
            count += 1
    return count


def serialize_result(result: Any, width: int, height: int) -> list[dict[str, Any]]:
    hands = []
    handedness = list(getattr(result, "handedness", []))
    world_sets = list(getattr(result, "hand_world_landmarks", []))
    for index, landmarks in enumerate(getattr(result, "hand_landmarks", [])):
        normalized = [
            {"x": float(point.x), "y": float(point.y), "z": float(point.z)}
            for point in landmarks
        ]
        wrist = np.asarray(
            [normalized[0]["x"], normalized[0]["y"], normalized[0]["z"]],
            dtype=np.float64,
        )
        knuckle_center = np.mean(
            [
                [normalized[i]["x"], normalized[i]["y"], normalized[i]["z"]]
                for i in PALM_KNUCKLE_LANDMARKS
            ],
            axis=0,
        )
        palm = (wrist + knuckle_center) / 2.0
        category = handedness[index][0] if index < len(handedness) and handedness[index] else None
        world = world_sets[index] if index < len(world_sets) else []
        finger_count = extended_finger_count(normalized)
        hands.append(
            {
                "index": index,
                "handedness": (
                    str(_category_value(category, "category_name", "unknown"))
                    if category is not None
                    else "unknown"
                ),
                "handedness_score": (
                    float(_category_value(category, "score", 0.0))
                    if category is not None
                    else 0.0
                ),
                "palm_center_normalized": palm.tolist(),
                "palm_center_pixel": [
                    int(round(float(palm[0]) * width)),
                    int(round(float(palm[1]) * height)),
                ],
                "extended_finger_count": finger_count,
                "is_open_palm": finger_count == len(FINGER_LANDMARKS),
                "landmarks": normalized,
                "world_landmarks_m": [
                    {"x": float(point.x), "y": float(point.y), "z": float(point.z)}
                    for point in world
                ],
            }
        )
    return hands


def annotate_frame(frame: np.ndarray, hands: list[dict[str, Any]]) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for hand in hands:
        pixels = [
            (
                int(round(point["x"] * width)),
                int(round(point["y"] * height)),
            )
            for point in hand["landmarks"]
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(annotated, pixels[start], pixels[end], (0, 220, 0), 2)
        for pixel in pixels:
            cv2.circle(annotated, pixel, 3, (0, 255, 255), -1)
        palm = tuple(hand["palm_center_pixel"])
        cv2.drawMarker(
            annotated,
            palm,
            (255, 0, 255),
            cv2.MARKER_CROSS,
            22,
            2,
        )
        gesture = "open palm" if hand.get("is_open_palm") else "hand"
        label = (
            f"{hand['handedness']} {gesture} "
            f"{hand['handedness_score']:.2f}"
        )
        cv2.putText(
            annotated,
            label,
            (max(0, palm[0] - 50), max(22, palm[1] - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    status = f"MediaPipe Hand Landmarker - hands: {len(hands)}"
    cv2.rectangle(annotated, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        status,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 0),
        1,
        cv2.LINE_AA,
    )
    return annotated
