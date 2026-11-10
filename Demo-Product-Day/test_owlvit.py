#!/usr/bin/env python3
"""Test camera/detection orchestration without importing robot control."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any
from vision_client import DEFAULT_DETECTION_URL, VisionClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", help="candy class or alias")
    parser.add_argument("--vision-url", default="http://127.0.0.1:8094")
    parser.add_argument("--detection-url", default=DEFAULT_DETECTION_URL)
    parser.add_argument(
        "--legacy-capture", action="store_true",
        help="use the alternative OWL-ViT service's combined /api/capture endpoint",
    )
    parser.add_argument("--slot", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.1, 0.05],
        help="confidence thresholds to test in descending order",
    )
    return parser.parse_args()


def capture(
    base_url: str,
    slot: int,
    prompt: str,
    confidence: float,
    *,
    detection_url: str = DEFAULT_DETECTION_URL,
    legacy_capture: bool = False,
) -> dict[str, Any]:
    if not legacy_capture:
        return VisionClient(base_url, detection_url=detection_url).capture(
            slot, prompt, confidence
        )
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/capture",
        data=json.dumps(
            {
                "slot": slot,
                "prompt": prompt,
                "confidence": confidence,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vision service returned HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"vision service request failed: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("vision service response was not a JSON object")
    return result


def main() -> int:
    args = parse_args()
    prompt = args.object.strip()
    if not prompt:
        print("ERROR: object prompt cannot be empty", file=sys.stderr)
        return 2
    if any(threshold <= 0 or threshold >= 1 for threshold in args.thresholds):
        print("ERROR: thresholds must be between zero and one", file=sys.stderr)
        return 2

    detected = False
    try:
        for threshold in args.thresholds:
            result = capture(
                args.vision_url, args.slot, prompt, threshold,
                detection_url=args.detection_url, legacy_capture=args.legacy_capture,
            )
            status = result.get("status")
            if status == "detected":
                detected = True
                print(
                    f"threshold={threshold:.3f}: DETECTED "
                    f"score={float(result['score']):.3f}, "
                    f"box={result['box_xyxy']}, "
                    f"camera={result['camera_point_mm']}mm"
                )
            else:
                print(f"threshold={threshold:.3f}: {status}")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Annotated frame: {args.vision_url.rstrip('/')}/snapshots/{args.slot}.jpg")
    return 0 if detected else 1


if __name__ == "__main__":
    raise SystemExit(main())
