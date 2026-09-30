"""Entry point: ``py -3.12 -m sahaay`` (or run.bat)."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(prog="sahaay", description="Offline accessibility copilot for Snapdragon PCs")
    ap.add_argument("--no-vlm", action="store_true", help="do not load the vision-language model")
    ap.add_argument("--no-head", action="store_true", help="start with the head cursor paused")
    ap.add_argument("--no-voice", action="store_true", help="start with voice off")
    ap.add_argument("--lang", choices=["en", "hi"], help="speech recognition language hint")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--no-preview", action="store_true", help="hide the webcam preview window")
    ap.add_argument("--port", help="COM port of an Arduino switch interface (default: auto-detect)")
    args = ap.parse_args()

    from .app import Sahaay
    from .config import Settings

    s = Settings.load()
    if args.no_vlm:
        s.load_vlm = False
    if args.no_head:
        s.head_cursor = False
    if args.no_voice:
        s.voice = False
    if args.lang:
        s.language = args.lang
    if args.no_hud:
        s.hud = False
    if args.no_preview:
        s.camera_preview = False
    if args.port:
        s.bridge_port = args.port
    Sahaay(s).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
