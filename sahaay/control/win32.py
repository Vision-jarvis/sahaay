"""Thin Win32 wrappers for cursor and input injection (ctypes, no extra dependencies)."""
from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
MOUSEEVENTF_WHEEL = 0x0800


def screen_size() -> tuple[int, int]:
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def get_cursor() -> tuple[int, int]:
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def set_cursor(x: float, y: float) -> None:
    w, h = screen_size()
    user32.SetCursorPos(int(min(max(x, 0), w - 1)), int(min(max(y, 0), h - 1)))


def mouse_down(button: str = "left") -> None:
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN if button == "left" else MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)


def mouse_up(button: str = "left") -> None:
    user32.mouse_event(MOUSEEVENTF_LEFTUP if button == "left" else MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)


def click(button: str = "left", count: int = 1) -> None:
    for _ in range(count):
        mouse_down(button)
        mouse_up(button)


def scroll(ticks: int) -> None:
    user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(ticks * 120), 0)
