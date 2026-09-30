"""Render the README infographics (banner, stats) and export chosen deck slides as README figures.

    py -3.12 docs/readme/render_readme.py

Needs the deck rendered first (docs/deck/render.py). Writes docs/media/readme_*.png.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MEDIA = ROOT / "docs" / "media"
sys.path.insert(0, str(ROOT / "docs" / "deck"))
from render import browser  # noqa: E402

PAGES = {"banner": (1600, 560), "stats": (1600, 290)}
SLIDES = {"06b": "readme_live", "07": "readme_architecture", "08": "readme_benchmarks", "10": "readme_cloud_vs_npu", "11": "readme_challenges", "06": "readme_storyboard"}


def shoot(html: Path, png: Path, w: int, h: int) -> None:
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h}",
                    "--force-device-scale-factor=2", "--virtual-time-budget=4000", f"--screenshot={png}", html.resolve().as_uri()],
                   check=True, capture_output=True, timeout=120)


def main() -> int:
    MEDIA.mkdir(parents=True, exist_ok=True)
    for name, (w, h) in PAGES.items():
        out = MEDIA / f"readme_{name}.png"
        shoot(HERE / f"{name}.html", out, w, h)
        Image.open(out).convert("RGB").save(out, optimize=True)
        print("wrote", out.name)
    out_dir = ROOT / "docs" / "deck" / "out"
    by_order = {f.stem: f for f in out_dir.glob("*.png")}
    for num, name in SLIDES.items():
        src = by_order.get(num)
        if src is None:
            continue
        im = Image.open(src).convert("RGB").resize((1920, 1080), Image.LANCZOS)
        im.save(MEDIA / f"{name}.png", optimize=True)
        print("wrote", name + ".png", "from", src.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
