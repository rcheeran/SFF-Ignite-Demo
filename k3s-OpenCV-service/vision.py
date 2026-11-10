"""Camera-local depth and annotation helpers; no detection client or model."""

from base_vision import (  # noqa: F401
    VisionError,
    annotate_capture,
    camera_point_from_pixel,
    robust_box_depth,
    robust_pixel_depth,
)
