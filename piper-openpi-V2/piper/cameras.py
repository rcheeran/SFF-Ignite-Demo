"""Thread-per-camera grabbers returning the latest 480x640 BGR frame."""
import threading
import time

import cv2
import numpy as np

from piper import config as C


class _Grabber:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._n = 0
        self._t0 = time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _put(self, frame):
        with self._lock:
            self._frame = frame
            self._n += 1

    def read(self) -> np.ndarray:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def wait(self, timeout=5.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.read() is not None:
                return True
            time.sleep(0.05)
        return False

    def fps(self) -> float:
        return self._n / max(time.time() - self._t0, 1e-6)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


class UsbCamera(_Grabber):
    def __init__(self, dev, width=C.IMG_W, height=C.IMG_H, fps=C.CAM_FPS):
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {dev}")
        super().__init__()

    def _loop(self):
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if ok:
                if f.shape[:2] != (C.IMG_H, C.IMG_W):
                    f = cv2.resize(f, (C.IMG_W, C.IMG_H))
                self._put(f)
            else:
                time.sleep(0.02)
        self.cap.release()


class RealSenseCamera(_Grabber):
    def __init__(self, serial=C.WRIST_CAM_SERIAL, width=C.IMG_W, height=C.IMG_H, fps=C.CAM_FPS):
        import pyrealsense2 as rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(str(serial))
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipe.start(cfg)
        super().__init__()

    def _loop(self):
        while not self._stop.is_set():
            try:
                fr = self.pipe.wait_for_frames(2000)
                self._put(np.asanyarray(fr.get_color_frame().get_data()).copy())
            except RuntimeError:
                time.sleep(0.05)
        self.pipe.stop()


def open_cameras(names=("top", "wrist"), top_dev=C.TOP_CAM_DEV) -> dict:
    ctor = {"top": lambda: UsbCamera(top_dev), "wrist": RealSenseCamera}
    cams = {name: ctor[name]() for name in names}
    for name, cam in cams.items():
        if not cam.wait():
            raise RuntimeError(f"no frames from camera '{name}'")
    return cams


class FakeCamera:
    """Synthetic frames (moving square) for --fake runs without hardware."""
    def __init__(self, name):
        self.name, self._t0 = name, time.time()

    def read(self):
        f = np.full((C.IMG_H, C.IMG_W, 3), 40, np.uint8)
        x = int((C.IMG_W - 80) * (0.5 + 0.5 * np.sin(time.time() - self._t0)))
        cv2.rectangle(f, (x, 200), (x + 80, 280), (0, 200, 255), -1)
        cv2.putText(f, self.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return f

    def fps(self):
        return float(C.CAM_FPS)

    def close(self):
        pass
