#!/usr/bin/env python3
"""Independent camera, depth, localization, and streaming service."""

from __future__ import annotations

import html

import base_app

_base_page_html = base_app.page_html


def page_html() -> bytes:
    page = (
        _base_page_html()
        .replace(b"Piper OWL-ViT vision", b"Piper Camera View")
        .replace(b"Piper D405 / OWL-ViT service", b"Camera View")
        .replace(b"Latest object scan", b"Latest object scan (published by demo)")
        .replace(b"Live D405 stream", b"Live RealSense stream")
    )
    if base_app.env_bool("CAMERA_PREVIEW_ONLY", False):
        page = page.replace(
            b"<main>",
            b"<main><p style='color:#ffcc00'>PREVIEW ONLY: camera images are available. "
            b"Depth/XYZ localization and workspace overlays are disabled until this "
            b"camera's calibration is accepted.</p>",
        )
    return page


base_app.page_html = page_html


class VisionApplication(base_app.VisionApplication):
    detection_enabled = False
    service_name = "opencv-vision"

    def __init__(self) -> None:
        super().__init__()
        self.camera.show_status_banner = False

    def dashboard_html(self) -> bytes:
        camera_name = html.escape(self.camera.name or "Camera identity unavailable")
        service_name = html.escape(self.service_name)
        title = f"Camera View — {camera_name} | {service_name}".encode("utf-8")
        return page_html().replace(b"<h2>Camera View</h2>", b"<h2>" + title + b"</h2>", 1)


create_server = base_app.create_server


def main() -> int:
    port = base_app.env_int("PORT", 8094)
    app = VisionApplication()
    try:
        app.start()
        server = create_server(app, port)
        print(f"OpenCV camera UI listening on 0.0.0.0:{port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping OpenCV vision service", flush=True)
    finally:
        app.camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
