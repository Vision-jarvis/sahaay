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
    Sahaay(s).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
