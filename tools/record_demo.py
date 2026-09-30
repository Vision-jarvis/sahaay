"""Record a demo clip of the screen (plus optional webcam inset) and turn it into README/deck assets.

    py -3.12 tools/record_demo.py <name> [seconds]     e.g.  handsfree 60  |  narrator 60

Press Ctrl+Alt+R to stop early. Writes to docs/media/:
  <name>.mp4          full-quality clip (ffmpeg, screen at 30 fps)
  <name>.gif          960 px wide, 12 fps, for the README
  <name>_still_N.png  six evenly spaced stills; pick the best for the deck
Only captures the primary monitor. Nothing is uploaded anywhere.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
FFMPEG = shutil.which("ffmpeg") or str(next(Path.home().glob(
    "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"), "ffmpeg"))


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "demo"
    secs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    MEDIA.mkdir(parents=True, exist_ok=True)
    mp4 = MEDIA / f"{name}.mp4"
    print(f"Recording {name} for up to {secs}s in 3 s ... (Ctrl+Alt+R to stop)")
    time.sleep(3)
    proc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "gdigrab", "-framerate", "30", "-draw_mouse", "1",
                             "-i", "desktop", "-t", str(secs), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                             "-pix_fmt", "yuv420p", str(mp4)], stdin=subprocess.PIPE)
    try:
        from pynput import keyboard

        def stop():
            try:
                proc.stdin.write(b"q")
                proc.stdin.flush()
            except Exception:
                pass

        hk = keyboard.GlobalHotKeys({"<ctrl>+<alt>+r": stop})
        hk.start()
    except Exception:
        hk = None
    proc.wait()
    if hk:
        hk.stop()
    print("saved", mp4)
    # GIF for the README (palette pass for quality)
    gif = MEDIA / f"{name}.gif"
    vf = "fps=12,scale=960:-1:flags=lanczos"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                    f"{vf},split[a][b];[a]palettegen=max_colors=160[p];[b][p]paletteuse=dither=bayer:bayer_scale=4", str(gif)], check=False)
    print("saved", gif)
    # six stills for the deck
    dur = float(subprocess.run([FFMPEG.replace("ffmpeg.exe", "ffprobe.exe") if FFMPEG.endswith(".exe") else "ffprobe",
                                "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp4)],
                               capture_output=True, text=True).stdout.strip() or secs)
    for i in range(6):
        t = dur * (i + 0.5) / 6
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(mp4), "-frames:v", "1",
                        str(MEDIA / f"{name}_still_{i + 1}.png")], check=False)
    print(f"saved 6 stills: {name}_still_1..6.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
