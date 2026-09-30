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

W, H = 560, 420
GREEN = (61, 220, 151)
SAFFRON = (232, 99, 43)


class CameraPreview:
    def __init__(self, root: tk.Tk, source: Callable[[], tuple]) -> None:
        self.source = source
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        sw, sh = win32.screen_size()
        self.win.geometry(f"{W}x{H + 40}+{sw - W - 24}+{sh - H - 40 - 72}")
        self.win.configure(bg="#101418")
        self.label = tk.Label(self.win, bg="#101418", bd=0)
        self.label.pack()
        self.caption = tk.Label(self.win, text="", fg="#8A97A6", bg="#101418", font=("Segoe UI Semibold", 14), anchor="w")
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
                # crop a window around the face (2.4x the face box) so the face fills the preview
                tgt = (0.0, 0.0, float(fw), float(fh))
                if face is not None and getattr(face, 'box', None):
                    bx0, by0, bx1, by1 = face.box
                    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
                    ch = min(fh, max(200.0, (by1 - by0) * 1.9))
                    cw = min(fw, ch * W / H)
                    tgt = (min(max(cx - cw / 2, 0), fw - cw), min(max(cy - ch / 2, 0), fh - ch), cw, ch)
                prev = getattr(self, '_crop', tgt)
                self._crop = tuple(0.8 * a + 0.2 * b for a, b in zip(prev, tgt))
                x0, y0, cw, ch = self._crop
                img = Image.fromarray(frame).crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((W, H), Image.BILINEAR)
                sx, sy = W / cw, H / ch
                d = ImageDraw.Draw(img)
                if face is not None and getattr(face, "found", False) and face.landmarks is not None:
                    pts = (face.landmarks[:, :2] - np.array([x0, y0])) * np.array([sx, sy])
                    for x, y in pts:
                        d.ellipse((float(x) - 1.2, float(y) - 1.2, float(x) + 1.2, float(y) + 1.2), fill=GREEN)
                    nose = pts[1]
                    d.line([tuple(map(float, nose)), (float(nose[0] + face.yaw * 120), float(nose[1] + face.pitch * 120))], fill=SAFFRON, width=5)
                    ms = face.detector_ms + face.landmark_ms
                    self.caption.config(text=f"face mesh on the Hexagon NPU · {ms:.1f} ms", fg="#3DDC97")
                else:
                    self.caption.config(text="looking for a face ...", fg="#8A97A6")
                img = img.transpose(Image.FLIP_LEFT_RIGHT)  # mirror, like a video call
                self._img = ImageTk.PhotoImage(img)
                self.label.config(image=self._img)
        except Exception:
            pass
        self.win.after(66, self._tick)
