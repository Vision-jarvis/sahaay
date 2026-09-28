"""Webcam capture via DirectShow (pygrabber). Returns RGB uint8 frames.

OpenCV has no ARM64 wheel for Windows on Snapdragon, so we talk to DirectShow through
comtypes. Frames arrive on a callback thread; ``read`` hands out the latest one.
"""
from __future__ import annotations

import threading
import time

import numpy as np


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        import comtypes

        try:
            comtypes.CoInitialize()  # DirectShow is COM; worker threads must initialise it
        except OSError:
            pass
        from pygrabber.dshow_graph import FilterGraph

        self._g = FilterGraph()
        self.devices = self._g.get_input_devices()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._g.add_video_input_device(index)
        try:
            dev = self._g.get_input_device()
            fmts = dev.get_formats()
            best = None
            for f in fmts:
                if f["width"] == width and f["height"] == height:
                    best = f
                    break
            if best is not None:
                dev.set_format(best["index"])
        except Exception:
            pass  # keep the default format
        self._g.add_sample_grabber(self._on_frame)
        self._g.add_null_render()
        self._g.prepare_preview_graph()
        self._running = False

    def _on_frame(self, img: np.ndarray) -> None:
        # pygrabber delivers BGR; flip to RGB once here
        with self._lock:
            self._frame = img[:, :, ::-1]
            self._stamp = time.time()

    def start(self) -> None:
        self._g.run()
        self._running = True

    def stop(self) -> None:
        if self._running:
            self._g.stop()
            self._running = False

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Return the newest frame (RGB, HxWx3 uint8), waiting up to ``timeout`` for a new one."""
        t_end = time.time() + timeout
        seen = self._stamp
        while time.time() < t_end:
            self._g.grab_frame()
            with self._lock:
                if self._frame is not None and self._stamp != seen:
                    return np.ascontiguousarray(self._frame)
            time.sleep(0.004)
        with self._lock:
            return None if self._frame is None else np.ascontiguousarray(self._frame)
