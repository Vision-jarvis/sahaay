"""Headless face-tracker check: grab webcam frames, run the NPU face pipeline, print pose and
timing, and save an annotated frame. Usage: py -3.12 tools/face_demo.py [seconds] [out.png]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sahaay.vision.camera import Camera  # noqa: E402
from sahaay.vision.face import FaceTracker  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "aihub" / "mediapipe_face_x2" / "mediapipe_face-onnx-float"


def main() -> int:
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("face_demo.png")
    tracker = FaceTracker(MODEL_DIR)
    print("detector providers:", tracker.det_info.providers, "| landmark providers:", tracker.lmk_info.providers)
    cam = Camera(width=1280, height=720)
    cam.start()
    t_end = time.time() + secs
    n = found = 0
    last = None
    det_ms = lmk_ms = 0.0
    frame = None
    try:
        while time.time() < t_end:
            frame = cam.read(timeout=1.0)
            if frame is None:
                continue
            r = tracker.process(frame)
            n += 1
            det_ms += r.detector_ms
            lmk_ms += r.landmark_ms
            if r.found:
                found += 1
                last = (r, frame)
                if found % 10 == 1:
                    print(f"score {r.score:.2f} yaw {r.yaw:+.2f} pitch {r.pitch:+.2f} roll {r.roll:+.2f} "
                          f"earL {r.ear_left:.2f} earR {r.ear_right:.2f} mouth {r.mouth_open:.2f} "
                          f"| det {r.detector_ms:.2f} ms lmk {r.landmark_ms:.2f} ms")
    finally:
        cam.stop()
    print(f"frames {n} in {secs}s ({n/secs:.1f} fps), faces found {found}, avg det {det_ms/max(n,1):.2f} ms, avg lmk {lmk_ms/max(n,1):.2f} ms")
    if last is not None:
        r, frame = last
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        for x, y, _ in r.landmarks:
            d.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 255, 0))
        if r.roi is not None:
            pts = [tuple(p) for p in r.roi[[0, 1, 3, 2]]]
            d.polygon(pts, outline=(255, 0, 0))
        if r.box is not None:
            d.rectangle(r.box, outline=(0, 128, 255))
        d.text((10, 10), f"yaw {r.yaw:+.2f} pitch {r.pitch:+.2f} roll {r.roll:+.2f} earL {r.ear_left:.2f} earR {r.ear_right:.2f} mouth {r.mouth_open:.2f}", fill=(255, 255, 0))
        img.save(out)
        print("saved", out)
    elif frame is not None:
        Image.fromarray(frame).save(out)
        print("no face; saved raw frame", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
