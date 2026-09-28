"""Generate the deck's vector assets from real data (no AI-generated imagery).

- img/face_mesh.svg: the MediaPipe canonical 468-point face mesh (the exact topology Sahaay runs on
  the NPU), rotated to a three-quarter view, triangulation as hairlines, feature contours brighter.
- img/hexagons.svg: concentric hexagon field, the Hexagon NPU motif.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
RED = "#E3261D"


def face_mesh(out: Path, yaw_deg: float = 28.0, pitch_deg: float = -6.0, size: int = 1000) -> None:
    d = json.loads((SRC / "data" / "face_mesh.json").read_text())
    V = d["vertices"]
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    pts = []
    for x, y, z in V:
        # yaw around the vertical axis, then a little pitch
        x1, z1 = x * cy + z * sy, -x * sy + z * cy
        y1, z2 = y * cp - z1 * sp, y * sp + z1 * cp
        pts.append((x1, y1, z2))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    scale = (size * 0.86) / max(maxx - minx, maxy - miny)
    ox = (size - (maxx - minx) * scale) / 2
    oy = (size - (maxy - miny) * scale) / 2

    def P(i):
        x, y, z = pts[i]
        return (ox + (x - minx) * scale, size - (oy + (y - miny) * scale), z)

    zs = [p[2] for p in pts]
    zmin, zmax = min(zs), max(zs)
    lines = []
    for a, b, c in d["faces"]:
        for i, j in ((a, b), (b, c), (c, a)):
            if i < j:
                x1, y1, z1 = P(i)
                x2, y2, z2 = P(j)
                depth = ((z1 + z2) / 2 - zmin) / max(zmax - zmin, 1e-6)  # 1 = closest to viewer
                op = 0.10 + 0.35 * depth
                lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke-opacity="{op:.2f}"/>')
    contours = []
    for i, j in d["connections"]:
        x1, y1, _ = P(i)
        x2, y2, _ = P(j)
        contours.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
    dots = []
    for i in range(len(pts)):
        x, y, z = P(i)
        depth = (z - zmin) / max(zmax - zmin, 1e-6)
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{1.6 + 1.4 * depth:.1f}" fill-opacity="{0.35 + 0.6 * depth:.2f}"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}">'
           f'<g stroke="{RED}" stroke-width="0.8" fill="none">{"".join(lines)}</g>'
           f'<g stroke="#F2F2EF" stroke-width="1.6" stroke-opacity="0.85" fill="none" stroke-linecap="round">{"".join(contours)}</g>'
           f'<g fill="#F2F2EF">{"".join(dots)}</g></svg>')
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out.name}: {len(lines)} edges, {len(contours)} contour segments, {len(dots)} points")


def hexagons(out: Path, size: int = 1000, rings: int = 9) -> None:
    cx, cy = size / 2, size / 2
    parts = []
    for r in range(1, rings + 1):
        rad = r * (size * 0.48 / rings)
        pts = " ".join(f"{cx + rad * math.cos(math.radians(60 * k - 30)):.1f},{cy + rad * math.sin(math.radians(60 * k - 30)):.1f}" for k in range(6))
        op = 0.9 - 0.085 * r
        parts.append(f'<polygon points="{pts}" stroke-opacity="{op:.2f}"/>')
    core = " ".join(f"{cx + 34 * math.cos(math.radians(60 * k - 30)):.1f},{cy + 34 * math.sin(math.radians(60 * k - 30)):.1f}" for k in range(6))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}">'
           f'<g fill="none" stroke="{RED}" stroke-width="1.5">{"".join(parts)}</g>'
           f'<polygon points="{core}" fill="{RED}"/></svg>')
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    face_mesh(SRC / "img" / "face_mesh.svg")
    hexagons(SRC / "img" / "hexagons.svg")
