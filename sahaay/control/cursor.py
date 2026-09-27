"""Head cursor: turns head pose from the NPU face tracker into cursor motion and clicks.

Joystick mode: yaw/pitch beyond a dead zone drive cursor velocity (non-linear gain), so the
user keeps a comfortable neutral pose and nudges the head to move. Clicks come from dwell
(hold still), blinks (short blink = left click, long blink = right click) and mouth-open
(toggle drag). All thresholds are user-tunable and captured by a calibration step.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..vision.face import FaceResult
from . import win32


@dataclass
class CursorConfig:
    dead_zone: float = 0.06        # normalised head-pose units ignored around neutral
    gain: float = 1800.0           # px/s at one unit beyond the dead zone
    expo: float = 1.6              # non-linearity: small nudges are precise, big ones fast
    smoothing: float = 0.35        # EMA on pose (0 = raw, 0.9 = very smooth)
    invert_y: bool = False
    dwell_enabled: bool = True
    dwell_time: float = 0.9        # s of stillness before a dwell click
    dwell_radius: float = 14.0     # px of allowed drift while dwelling
    dwell_cooldown: float = 1.2    # s before another dwell click at the same spot
    blink_enabled: bool = True
    ear_threshold: float = 0.17    # eye aspect ratio below this = eyes closed
    blink_min: float = 0.10        # s; shorter is noise
    blink_click_max: float = 0.45  # s; a blink up to this long = left click
    blink_right_max: float = 1.2   # s; up to this long = right click
    mouth_threshold: float = 0.42  # lip gap / mouth width to count as "open"
    mouth_hold: float = 0.35       # s open before drag toggles


@dataclass
class CursorState:
    enabled: bool = True
    neutral_yaw: float = 0.0
    neutral_pitch: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    last_event: str = ""
    dragging: bool = False
    _eyes_closed_since: float | None = None
    _mouth_open_since: float | None = None
    _mouth_latched: bool = False
    _dwell_anchor: tuple[float, float] | None = None
    _dwell_since: float = 0.0
    _last_dwell_click: float = 0.0
    _last_dwell_pos: tuple[float, float] = (-1e9, -1e9)
    _pos: tuple[float, float] = field(default_factory=lambda: tuple(map(float, win32.get_cursor())))
    _last_t: float = field(default_factory=time.perf_counter)


class HeadCursor:
    def __init__(self, cfg: CursorConfig | None = None) -> None:
        self.cfg = cfg or CursorConfig()
        self.st = CursorState()

    # ---- calibration ------------------------------------------------------------------
    def calibrate(self, yaw: float, pitch: float) -> None:
        self.st.neutral_yaw, self.st.neutral_pitch = yaw, pitch
        self.st.yaw = self.st.pitch = 0.0

    # ---- main update ------------------------------------------------------------------
    def update(self, r: FaceResult) -> str:
        """Feed one frame's face result. Returns a short event string ("" if nothing happened)."""
        now = time.perf_counter()
        dt = min(max(now - self.st._last_t, 1e-3), 0.1)
        self.st._last_t = now
        self.st.last_event = ""
        if not self.st.enabled or not r.found:
            self.st._eyes_closed_since = None
            return ""
        a = self.cfg.smoothing
        self.st.yaw = a * self.st.yaw + (1 - a) * (r.yaw - self.st.neutral_yaw)
        self.st.pitch = a * self.st.pitch + (1 - a) * (r.pitch - self.st.neutral_pitch)
        self._move(dt)
        self._gestures(r, now)
        self._dwell(now)
        return self.st.last_event

    def _axis(self, v: float) -> float:
        m = abs(v) - self.cfg.dead_zone
        if m <= 0:
            return 0.0
        return math.copysign(self.cfg.gain * (m ** self.cfg.expo), v)

    def _move(self, dt: float) -> None:
        vx = self._axis(self.st.yaw)
        vy = self._axis(self.st.pitch) * (-1 if self.cfg.invert_y else 1)
        if vx == 0 and vy == 0:
            # keep our notion of position in sync if the user also has a mouse
            self.st._pos = tuple(map(float, win32.get_cursor()))
            return
        x = self.st._pos[0] + vx * dt
        y = self.st._pos[1] + vy * dt
        w, h = win32.screen_size()
        x, y = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
        self.st._pos = (x, y)
        win32.set_cursor(x, y)

    def _gestures(self, r: FaceResult, now: float) -> None:
        c = self.cfg
        if c.blink_enabled:
            closed = r.ear_left < c.ear_threshold and r.ear_right < c.ear_threshold
            if closed and self.st._eyes_closed_since is None:
                self.st._eyes_closed_since = now
            elif not closed and self.st._eyes_closed_since is not None:
                dur = now - self.st._eyes_closed_since
                self.st._eyes_closed_since = None
                if c.blink_min <= dur <= c.blink_click_max:
                    win32.click("left")
                    self.st.last_event = "blink:left_click"
                elif c.blink_click_max < dur <= c.blink_right_max:
                    win32.click("right")
                    self.st.last_event = "blink:right_click"
        # mouth open toggles drag
        opened = r.mouth_open > c.mouth_threshold
        if opened:
            if self.st._mouth_open_since is None:
                self.st._mouth_open_since = now
            elif not self.st._mouth_latched and now - self.st._mouth_open_since >= c.mouth_hold:
                self.st._mouth_latched = True
                if self.st.dragging:
                    win32.mouse_up("left")
                    self.st.dragging = False
                    self.st.last_event = "mouth:drag_end"
                else:
                    win32.mouse_down("left")
                    self.st.dragging = True
                    self.st.last_event = "mouth:drag_start"
        else:
            self.st._mouth_open_since = None
            self.st._mouth_latched = False

    def _dwell(self, now: float) -> None:
        c = self.cfg
        if not c.dwell_enabled or self.st.dragging:
            return
        pos = self.st._pos
        anchor = self.st._dwell_anchor
        if anchor is None or math.hypot(pos[0] - anchor[0], pos[1] - anchor[1]) > c.dwell_radius:
            self.st._dwell_anchor = pos
            self.st._dwell_since = now
            return
        if now - self.st._dwell_since >= c.dwell_time:
            far_from_last = math.hypot(pos[0] - self.st._last_dwell_pos[0], pos[1] - self.st._last_dwell_pos[1]) > c.dwell_radius
            if far_from_last and now - self.st._last_dwell_click >= c.dwell_cooldown:
                win32.click("left")
                self.st._last_dwell_click = now
                self.st._last_dwell_pos = pos
                self.st.last_event = "dwell:left_click"
            self.st._dwell_since = now  # restart the timer either way

    @property
    def dwell_progress(self) -> float:
        """0..1 fraction of the dwell timer, for the HUD ring."""
        if not self.cfg.dwell_enabled or self.st._dwell_anchor is None:
            return 0.0
        return min(1.0, (time.perf_counter() - self.st._dwell_since) / self.cfg.dwell_time)
