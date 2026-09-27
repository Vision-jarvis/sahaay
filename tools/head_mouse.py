"""Drive the real Windows cursor with your head for N seconds (default 20).

    py -3.12 tools/head_mouse.py [seconds] [--no-clicks]

First second: hold a neutral pose while it calibrates. Then nudge left/right/up/down to move,
hold still to dwell-click, blink to click, open your mouth for a moment to start/stop dragging.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sahaay.control.cursor import CursorConfig, HeadCursor  # noqa: E402
from sahaay.vision.camera import Camera  # noqa: E402
from sahaay.vision.face import FaceTracker  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "aihub" / "mediapipe_face_x2" / "mediapipe_face-onnx-float"


def main() -> int:
    secs = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 20.0
    no_clicks = "--no-clicks" in sys.argv
    cfg = CursorConfig(dwell_enabled=not no_clicks, blink_enabled=not no_clicks)
    tracker = FaceTracker(MODEL_DIR)
    print("NPU:", tracker.on_npu)
    cursor = HeadCursor(cfg)
    cam = Camera(width=1280, height=720)
    cam.start()
    try:
        # calibration: average pose over ~1 s
        yaws, pitches = [], []
        t_end = time.time() + 1.0
        while time.time() < t_end:
            f = cam.read()
            if f is None:
                continue
            r = tracker.process(f)
            if r.found:
                yaws.append(r.yaw)
                pitches.append(r.pitch)
        if not yaws:
            print("no face during calibration")
            return 1
        cursor.calibrate(sum(yaws) / len(yaws), sum(pitches) / len(pitches))
        print(f"calibrated neutral yaw {cursor.st.neutral_yaw:+.3f} pitch {cursor.st.neutral_pitch:+.3f}. Go.")
        t_end = time.time() + secs
        n = 0
        events = []
        while time.time() < t_end:
            f = cam.read()
            if f is None:
                continue
            r = tracker.process(f)
            ev = cursor.update(r)
            n += 1
            if ev:
                events.append(ev)
                print(time.strftime("%H:%M:%S"), ev)
        print(f"frames {n} in {secs}s ({n/secs:.1f} fps); events: {events}")
    finally:
        cam.stop()
        if cursor.st.dragging:
            from sahaay.control import win32
            win32.mouse_up("left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
