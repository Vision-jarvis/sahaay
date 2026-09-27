"""Visual description with Qwen3-VL on the Hexagon NPU (GenieX, QAIRT runtime).

Used for what UI Automation cannot tell a blind user: photos, charts, canvases, and the
physical world through the webcam. Loaded lazily on a background thread; the first call
waits for it.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

SCREEN_PROMPT = ("Describe what is on this screen for a person who cannot see it. Be concrete and brief: two or three "
                 "short sentences, mention the application, the main content, and any button or field that looks important. "
                 "Do not guess details you cannot see.")
CAMERA_PROMPT = ("Describe what the camera sees for a person who cannot see. Two short sentences: the people, objects and "
                 "text in view, and anything that matters for safety. Do not guess.")


class Describer:
    def __init__(self, model_id: str = "ai-hub-models/Qwen3-VL-4B-Instruct", preload: bool = True, max_side: int = 1280) -> None:
        self.model_id = model_id
        self.max_side = max_side
        self._vlm = None
        self._lock = threading.Lock()
        self.ready = threading.Event()
        self.load_ms = 0.0
        self.last_ms = 0.0
        self.last_ttft_ms = 0.0
        self.last_tokens = 0
        self._tmp = Path(tempfile.gettempdir()) / "sahaay_vlm"
        self._tmp.mkdir(exist_ok=True)
        if preload:
            threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        from geniex import AutoModelForVision2Seq

        t0 = time.perf_counter()
        vlm = AutoModelForVision2Seq.from_pretrained(self.model_id)
        with self._lock:
            self._vlm = vlm
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.ready.set()

    # ---- inputs ----------------------------------------------------------------------------
    def _save(self, img: Image.Image, name: str) -> str:
        w, h = img.size
        s = self.max_side / max(w, h)
        if s < 1:
            img = img.resize((int(w * s), int(h * s)), Image.BILINEAR)
        p = self._tmp / f"{name}_{int(time.time()*1000)}.png"
        img.convert("RGB").save(p)
        return str(p)

    def screenshot(self, region: tuple[int, int, int, int] | None = None) -> Image.Image:
        import mss

        with mss.mss() as sct:
            mon = sct.monitors[1]
            if region:
                l, t, r, b = region
                mon = {"left": l, "top": t, "width": r - l, "height": b - t}
            shot = sct.grab(mon)
            return Image.frombytes("RGB", shot.size, shot.rgb)

    # ---- generation -------------------------------------------------------------------------
    def describe_image(self, img: Image.Image | np.ndarray, prompt: str, max_new_tokens: int = 120, timeout: float = 60.0) -> str:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if not self.ready.wait(timeout):
            return "The vision model is still loading. Please try again in a moment."
        path = self._save(img, "img")
        try:
            with self._lock:
                vlm = self._vlm
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                p = vlm.tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
                t0 = time.perf_counter()
                first = None
                parts = []
                for ch in vlm.generate(p, max_new_tokens=max_new_tokens, temperature=0.0, images=[path], stream=True):
                    if first is None:
                        first = time.perf_counter()
                    parts.append(ch if isinstance(ch, str) else str(ch))
                self.last_ms = (time.perf_counter() - t0) * 1000
                self.last_ttft_ms = ((first or t0) - t0) * 1000
                self.last_tokens = len(parts)
                try:
                    vlm.reset()
                except Exception:
                    pass
            return "".join(parts).strip()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def describe_screen(self, region: tuple[int, int, int, int] | None = None) -> str:
        return self.describe_image(self.screenshot(region), SCREEN_PROMPT)

    def describe_camera(self, frame_rgb: np.ndarray) -> str:
        return self.describe_image(frame_rgb, CAMERA_PROMPT)

    def close(self) -> None:
        with self._lock:
            if self._vlm is not None:
                self._vlm.close()
                self._vlm = None
