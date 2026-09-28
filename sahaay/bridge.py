"""Sahaay Bridge: physical switch access over USB serial (Arduino UNO Q, any Arduino, or a simulator).

Many people with severe motor impairment use one or two large buttons, or a sip-and-puff sensor,
rather than a camera. Any microcontroller that prints JSON lines over serial becomes a Sahaay
switch interface:

    {"switch": 1, "state": "down"}      left click (hold 600 ms = start/stop drag)
    {"switch": 2, "state": "down"}      right click
    {"switch": 3, "state": "down"}      push-to-talk: voice on while held
    {"switch": 4, "state": "down"}      pause / resume the head cursor
    {"joy": [x, y]}                     optional joystick, -1..1, moves the cursor

Protocol and a Wokwi-simulated sketch live in arduino/. Detected automatically when a COM port
answers a "hello" with a line containing "sahaay".
"""
from __future__ import annotations

import json
import threading
import time
from typing import Callable

from .control import win32


class SwitchBridge:
    def __init__(self, on_action: Callable[[str], None], port: str | None = None, baud: int = 115200) -> None:
        self.on_action = on_action
        self.port = port
        self.baud = baud
        self._stop = threading.Event()
        self._ser = None
        self.connected = False
        self._down_at: dict[int, float] = {}
        self._drag = False

    # ---- discovery -----------------------------------------------------------------------
    @staticmethod
    def find_port(baud: int = 115200, timeout: float = 1.5) -> str | None:
        try:
            import serial
            from serial.tools import list_ports
        except ImportError:
            return None
        for p in list_ports.comports():
            try:
                with serial.Serial(p.device, baud, timeout=timeout) as s:
                    time.sleep(1.8)  # Arduino resets on open
                    s.reset_input_buffer()
                    s.write(b'{"hello":"sahaay"}\n')
                    line = s.readline().decode("utf-8", "ignore")
                    if "sahaay" in line.lower():
                        return p.device
            except Exception:
                continue
        return None

    def start(self) -> bool:
        try:
            import serial
        except ImportError:
            return False
        port = self.port or self.find_port(self.baud)
        if not port:
            return False
        self._ser = serial.Serial(port, self.baud, timeout=0.2)
        self.connected = True
        threading.Thread(target=self._loop, daemon=True, name="bridge").start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass

    # ---- protocol ------------------------------------------------------------------------
    def handle_line(self, line: str) -> None:
        line = line.strip()
        if not line.startswith("{"):
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return
        if "joy" in msg:
            x, y = msg["joy"][:2]
            cx, cy = win32.get_cursor()
            win32.set_cursor(cx + float(x) * 18, cy + float(y) * 18)
            return
        sw = int(msg.get("switch", 0))
        state = msg.get("state")
        now = time.time()
        if state == "down":
            self._down_at[sw] = now
            if sw == 3:
                self.on_action("voice_on")
        elif state == "up":
            held = now - self._down_at.pop(sw, now)
            if sw == 1:
                if held >= 0.6:
                    if self._drag:
                        win32.mouse_up("left")
                        self._drag = False
                        self.on_action("drag_end")
                    else:
                        win32.mouse_down("left")
                        self._drag = True
                        self.on_action("drag_start")
                else:
                    win32.click("left")
                    self.on_action("left_click")
            elif sw == 2:
                win32.click("right")
                self.on_action("right_click")
            elif sw == 3:
                self.on_action("voice_off")
            elif sw == 4:
                self.on_action("toggle_head")

    def _loop(self) -> None:
        while not self._stop.is_set() and self._ser is not None:
            try:
                raw = self._ser.readline()
            except Exception:
                self.connected = False
                break
            if raw:
                self.handle_line(raw.decode("utf-8", "ignore"))
