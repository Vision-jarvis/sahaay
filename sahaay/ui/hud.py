"""Heads-up display: a small always-on-top panel plus a dwell ring that follows the cursor.

Tkinter only (ships with Python, works on ARM64). The panel shows what Sahaay heard, what it
did, which mode is on, and live per-model latency with the compute unit that ran it, so the
"which core runs what" story is visible to the user (and to a judge) at all times.
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, field
from typing import Callable

from ..control import win32

BG = "#101418"
FG = "#e8edf2"
DIM = "#8a97a6"
ACCENT = "#3ddc97"
WARN = "#ffb454"
KEY = "#ff00ff"  # transparent colour key for the ring window


@dataclass
class HudState:
    status: str = "starting"          # listening | hearing | thinking | speaking | paused
    heard: str = ""
    action: str = ""
    head_on: bool = True
    voice_on: bool = True
    face_found: bool = False
    dwell: float = 0.0
    mic_level: float = 0.0
    latencies: dict[str, tuple[float, str]] = field(default_factory=dict)  # name -> (ms, unit)
    note: str = ""


class Hud:
    def __init__(self, state: HudState, on_quit: Callable[[], None] | None = None) -> None:
        self.state = state
        self.on_quit = on_quit
        self.root = tk.Tk()
        self.root.withdraw()
        self.panel = tk.Toplevel(self.root)
        self.panel.overrideredirect(True)
        self.panel.attributes("-topmost", True)
        self.panel.attributes("-alpha", 0.92)
        self.panel.configure(bg=BG)
        w, h = 380, 176
        sw, _ = win32.screen_size()
        self.panel.geometry(f"{w}x{h}+{sw - w - 16}+16")
        f = ("Segoe UI", 10)
        fb = ("Segoe UI Semibold", 11)
        top = tk.Frame(self.panel, bg=BG)
        top.pack(fill="x", padx=10, pady=(8, 2))
        self.l_title = tk.Label(top, text="Sahaay", fg=ACCENT, bg=BG, font=("Segoe UI Semibold", 13))
        self.l_title.pack(side="left")
        self.l_status = tk.Label(top, text="", fg=FG, bg=BG, font=fb)
        self.l_status.pack(side="right")
        self.l_modes = tk.Label(self.panel, text="", fg=DIM, bg=BG, font=f, anchor="w")
        self.l_modes.pack(fill="x", padx=10)
        self.l_heard = tk.Label(self.panel, text="", fg=FG, bg=BG, font=f, anchor="w", wraplength=w - 20, justify="left")
        self.l_heard.pack(fill="x", padx=10, pady=(4, 0))
        self.l_action = tk.Label(self.panel, text="", fg=DIM, bg=BG, font=f, anchor="w", wraplength=w - 20, justify="left")
        self.l_action.pack(fill="x", padx=10)
        self.l_lat = tk.Label(self.panel, text="", fg=DIM, bg=BG, font=("Consolas", 9), anchor="w", justify="left")
        self.l_lat.pack(fill="x", padx=10, pady=(4, 6))
        self.panel.bind("<Button-1>", self._drag_start)
        self.panel.bind("<B1-Motion>", self._drag)
        # dwell ring window (colour-keyed transparency)
        self.ring = tk.Toplevel(self.root)
        self.ring.overrideredirect(True)
        self.ring.attributes("-topmost", True)
        self.ring.attributes("-transparentcolor", KEY)
        self.ring_size = 44
        self.ring.geometry(f"{self.ring_size}x{self.ring_size}+0+0")
        self.canvas = tk.Canvas(self.ring, width=self.ring_size, height=self.ring_size, bg=KEY, highlightthickness=0)
        self.canvas.pack()
        self._visible = True
        self._tick()

    def _drag_start(self, e):
        self._dx, self._dy = e.x, e.y

    def _drag(self, e):
        self.panel.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        (self.panel.deiconify if visible else self.panel.withdraw)()

    def _tick(self) -> None:
        s = self.state
        colour = {"listening": ACCENT, "hearing": ACCENT, "thinking": WARN, "speaking": "#7ec8ff", "paused": DIM}.get(s.status, FG)
        self.l_status.config(text=s.status.upper(), fg=colour)
        head = "Head " + ("●" if s.head_on and s.face_found else "○" if s.head_on else "off")
        voice = "Voice " + ("●" if s.voice_on else "off")
        lvl = "▁▂▃▄▅▆▇█"[min(7, int(s.mic_level * 300))] if s.voice_on else ""
        self.l_modes.config(text=f"{head}    {voice} {lvl}    {s.note}")
        self.l_heard.config(text=("Heard: " + s.heard) if s.heard else "Say a command, or nudge your head to move the cursor.")
        self.l_action.config(text=("Did: " + s.action) if s.action else "")
        if s.latencies:
            parts = [f"{k:<8}{v[0]:6.0f} ms  {v[1]}" for k, v in s.latencies.items()]
            self.l_lat.config(text="\n".join(parts))
        # dwell ring follows the cursor
        try:
            x, y = win32.get_cursor()
            r = self.ring_size
            self.ring.geometry(f"{r}x{r}+{x - r // 2}+{y - r // 2}")
            self.canvas.delete("all")
            if s.head_on and s.dwell > 0.05:
                self.canvas.create_arc(4, 4, r - 4, r - 4, start=90, extent=-360 * s.dwell, style="arc", outline=ACCENT, width=3)
            elif s.head_on and s.face_found:
                self.canvas.create_oval(r // 2 - 3, r // 2 - 3, r // 2 + 3, r // 2 + 3, outline=ACCENT, width=1)
        except Exception:
            pass
        self.root.after(60, self._tick)

    def run(self) -> None:
        self.root.mainloop()

    def quit(self) -> None:
        try:
            self.root.after(0, self.root.destroy)
        except Exception:
            pass
