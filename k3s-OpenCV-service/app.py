#!/usr/bin/env python3
"""D405 + OpenCV + Foundry YOLO service compatible with demo.py."""

from __future__ import annotations

import os

import base_app

_base_page_html = base_app.page_html


def page_html() -> bytes:
    return (
        _base_page_html()
        .replace(b"Piper OWL-ViT vision", b"Piper OpenCV / Foundry YOLO vision")
        .replace(b"Piper D405 / OWL-ViT service", b"Piper D405 / OpenCV + Foundry YOLO")
    )


base_app.page_html = page_html


class VisionApplication(base_app.VisionApplication):
    def __init__(self) -> None:
        super().__init__()
        camera_check = self.camera.check

        def check() -> None:
            camera_check()
            self.detector.check()

        self.camera.check = check


create_server = base_app.create_server


def main() -> int:
    port = base_app.env_int("PORT", 8094)
    app = VisionApplication()
    try:
        app.start()
        server = create_server(app, port)
        print(f"OpenCV/Foundry UI listening on 0.0.0.0:{port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping OpenCV vision service", flush=True)
    finally:
        app.camera.close()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OWL_MODEL", "candy-yolo11s")
    raise SystemExit(main())
