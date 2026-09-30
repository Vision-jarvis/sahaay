"""Corner webcam preview with the live NPU face mesh drawn on top.

Shows the user what Sahaay sees (and makes demos human): the latest camera frame, mirrored,
with the 468 landmarks from the NPU face tracker, the head-pose vector, and a small
"face tracked on NPU" label. Draggable; updates at ~15 fps from frames the camera loop
already captured, so it costs no extra camera access and no extra inference.
"""
from __future__ import annotations

import tkinter as tk
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from ..control import win32

W, H = 360, 270
GREEN = (61, 220, 151)
SAFFRON = (232, 99, 43)


class CameraPreview:
    def __init__(self, root: tk.Tk, source: Callable[[], tuple]) -> None:
        self.source = source
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        sw, sh = win32.screen_size()
        self.win.geometry(f"{W}x{H + 30}+{sw - W - 24}+{sh - H - 30 - 72}")
        self.win.configure(bg="#101418")
        self.label = tk.Label(self.win, bg="#101418", bd=0)
        self.label.pack()
        self.caption = tk.Label(self.win, text="", fg="#8A97A6", bg="#101418", font=("Segoe UI", 10), anchor="w")
        self.caption.pack(fill="x", padx=8)
        for w in (self.win, self.label, self.caption):
            w.bind("<Button-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag)
        self._img = None
        self._tick()

    def _drag_start(self, e):
        self._dx, self._dy = e.x, e.y

    def _drag(self, e):
        self.win.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _tick(self) -> None:
        try:
            frame, face = self.source()
            if frame is not None:
                fh, fw = frame.shape[:2]
                img = Image.fromarray(frame).resize((W, int(W * fh / fw)), Image.BILINEAR)
                sx, sy = W / fw, img.height / fh
                d = ImageDraw.Draw(img)
                if face is not None and getattr(face, "found", False) and face.landmarks is not None:
                    pts = face.landmarks[:, :2] * np.array([sx, sy])
                    for x, y in pts[::2]:
                        d.point((float(x), float(y)), fill=GREEN)
                    nose = pts[1]
                    d.line([tuple(nose), (float(nose[0] + face.yaw * 60), float(nose[1] + face.pitch * 60))], fill=SAFFRON, width=3)
                    ms = face.detector_ms + face.landmark_ms
                    self.caption.config(text=f"face mesh on the Hexagon NPU · {ms:.1f} ms", fg="#3DDC97")
                else:
                    self.caption.config(text="looking for a face ...", fg="#8A97A6")
                img = img.transpose(Image.FLIP_LEFT_RIGHT)  # mirror, like a video call
                img = img.crop((0, 0, W, min(H, img.height)))
                self._img = ImageTk.PhotoImage(img)
                self.label.config(image=self._img)
        except Exception:
            pass
        self.win.after(66, self._tick)
