"""Rendered stand-ins for piper.cameras: SimCamera('top' | 'wrist') -> latest 480x640 BGR frame of the twin.

    from sim.cameras import open_sim_cameras
    cams = open_sim_cameras()        # same dict / read() / fps() / close() as open_cameras(); connects the twin if needed

Each camera renders in its own thread (one EGL context per thread) at CAM_FPS, but only while something has
read it in the last second - teleop never reads cameras, so it pays nothing. VideoRecorder writes mp4s.
"""
import atexit
import os
import threading
import time

import cv2

from piper import config as C
from piper.cameras import _Grabber
from sim.piper_sim import connect_sim   # also sets MUJOCO_GL before mujoco is imported

import mujoco  # noqa: E402


class SimCamera(_Grabber):
    def __init__(self, name, sim=None):
        self.name, self.sim = name, sim or connect_sim()
        self._last_read = time.time()
        super().__init__()

    def read(self):
        self._last_read = time.time()
        return super().read()

    def _loop(self):
        r = mujoco.Renderer(self.sim.m, C.IMG_H, C.IMG_W)
        atexit.register(self.close)     # after the renderer: its GL backend registered its own teardown just now,
        try:                            # and atexit runs last-in-first-out (an unclosed camera would print a traceback)
            next_t = time.perf_counter()
            while not self._stop.is_set():
                if time.time() - self._last_read < 1.0:
                    with self.sim.lock:
                        r.update_scene(self.sim.d, camera=self.name)
                    self._put(r.render()[:, :, ::-1].copy())     # RGB -> BGR like the real cameras
                next_t = max(next_t + 1.0 / C.CAM_FPS, time.perf_counter() - 0.1)
                time.sleep(max(0.0, next_t - time.perf_counter()))
        finally:
            r.close()


def open_sim_cameras(names=tuple(C.CAMERA_NAMES), sim=None) -> dict:
    cams = {name: SimCamera(name, sim) for name in names}
    for name, cam in cams.items():
        if not cam.wait():
            raise RuntimeError(f"no frames from sim camera '{name}'")
    return cams


class VideoRecorder:
    """Writes each camera at FPS to <out_dir>/<prefix>_<name>.mp4 (cv2 mp4v) until close()."""
    def __init__(self, cams, out_dir=C.SIM_VIDEO_DIR, prefix="demo", fps=C.FPS):
        os.makedirs(out_dir, exist_ok=True)
        self.cams, self.fps, self.n = cams, fps, 0
        self.paths = {n: os.path.join(out_dir, f"{prefix}_{n}.mp4") for n in cams}
        self._w = {n: cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (C.IMG_W, C.IMG_H))
                   for n, p in self.paths.items()}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        next_t = time.monotonic()
        while not self._stop.is_set():
            for name, cam in self.cams.items():
                f = cam.read()
                if f is not None:
                    self._w[name].write(f)
            self.n += 1
            next_t += 1.0 / self.fps
            time.sleep(max(0.0, next_t - time.monotonic()))

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        for w in self._w.values():
            w.release()
        print(f"video: {self.n} frames ({self.n / self.fps:.0f} s) -> {' '.join(self.paths.values())}")
