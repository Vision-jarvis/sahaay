"""Action executor: carries out tool calls on Windows (CPU side of Sahaay)."""
from __future__ import annotations

import os
import subprocess
import time
from typing import Callable

from pynput.keyboard import Controller as KeyController
from pynput.keyboard import Key

from .. import screen as scr
from ..brain import ToolCall
from . import win32

APP_ALIASES = {
    "notepad": "notepad.exe", "calculator": "calc.exe", "calc": "calc.exe", "paint": "mspaint.exe",
    "explorer": "explorer.exe", "file explorer": "explorer.exe", "files": "explorer.exe", "my files": "explorer.exe",
    "downloads": os.path.join(os.path.expanduser("~"), "Downloads"), "documents": os.path.join(os.path.expanduser("~"), "Documents"),
    "desktop": os.path.join(os.path.expanduser("~"), "Desktop"),
    "settings": "ms-settings:", "control panel": "control.exe", "task manager": "taskmgr.exe", "terminal": "wt.exe",
    "command prompt": "cmd.exe", "cmd": "cmd.exe", "powershell": "powershell.exe",
    "edge": "msedge.exe", "microsoft edge": "msedge.exe", "browser": "msedge.exe", "chrome": "chrome.exe", "google chrome": "chrome.exe",
    "word": "winword.exe", "excel": "excel.exe", "powerpoint": "powerpnt.exe", "outlook": "outlook.exe",
    "vs code": "code", "vscode": "code", "visual studio code": "code", "spotify": "spotify.exe", "whatsapp": "whatsapp:",
    "camera": "microsoft.windows.camera:", "mail": "outlookmail:", "calendar": "outlookcal:", "photos": "ms-photos:",
    "store": "ms-windows-store:", "youtube": "https://www.youtube.com", "google": "https://www.google.com",
}

KEY_MAP = {
    "enter": Key.enter, "return": Key.enter, "escape": Key.esc, "esc": Key.esc, "tab": Key.tab, "space": Key.space,
    "backspace": Key.backspace, "delete": Key.delete, "del": Key.delete, "home": Key.home, "end": Key.end,
    "pageup": Key.page_up, "pagedown": Key.page_down, "up": Key.up, "down": Key.down, "left": Key.left, "right": Key.right,
    "ctrl": Key.ctrl, "control": Key.ctrl, "alt": Key.alt, "shift": Key.shift, "win": Key.cmd, "windows": Key.cmd, "cmd": Key.cmd,
    "f1": Key.f1, "f2": Key.f2, "f3": Key.f3, "f4": Key.f4, "f5": Key.f5, "f11": Key.f11, "f12": Key.f12,
    "volumeup": Key.media_volume_up, "volumedown": Key.media_volume_down, "mute": Key.media_volume_mute,
    "playpause": Key.media_play_pause, "capslock": Key.caps_lock, "printscreen": Key.print_screen,
}


class Executor:
    """Executes ToolCalls. Callbacks let the app speak, describe, and drive the cursor."""

    def __init__(self, *, say: Callable[[str], None], describe_screen: Callable[[], str] | None = None,
                 describe_camera: Callable[[], str] | None = None, cursor_action: Callable[[str], None] | None = None,
                 summarise: Callable[[str], str] | None = None) -> None:
        self.say = say
        self.describe_screen = describe_screen
        self.describe_camera = describe_camera
        self.cursor_action = cursor_action
        self.summarise = summarise
        self.kb = KeyController()
        self.last_snapshot: scr.Snapshot | None = None

    # ---- entry -----------------------------------------------------------------------------
    def run(self, calls: list[ToolCall]) -> list[str]:
        results = []
        for c in calls:
            fn = getattr(self, f"do_{c.tool}", None)
            if fn is None:
                results.append(f"unknown tool {c.tool}")
                continue
            try:
                results.append(fn(**c.args) or "ok")
            except TypeError as exc:
                results.append(f"bad args for {c.tool}: {exc}")
            except Exception as exc:
                results.append(f"{c.tool} failed: {exc}")
            time.sleep(0.15)
        return results

    # ---- tools -----------------------------------------------------------------------------
    @staticmethod
    def _top_windows() -> list:
        import uiautomation as auto

        try:
            return [w for w in auto.GetRootControl().GetChildren() if w.Name]
        except Exception:
            return []

    def _activate_new_window(self, before: set[int], hint: str, timeout: float = 4.0) -> str | None:
        """Wait for a window that was not there before (or whose title matches the hint) and bring it to front."""
        import uiautomation as auto

        t_end = time.time() + timeout
        while time.time() < t_end:
            for w in self._top_windows():
                h = w.NativeWindowHandle
                if h and (h not in before or hint and hint in (w.Name or "").lower()):
                    try:
                        w.SetActive()
                        w.SetFocus()
                    except Exception:
                        pass
                    if h not in before:
                        return w.Name
            time.sleep(0.15)
        return None

    def do_open_app(self, name: str = "", **_) -> str:
        key = name.strip().lower()
        for prefix in ("my ", "the ", "a "):
            if key.startswith(prefix):
                key = key[len(prefix):]
        target = APP_ALIASES.get(key)
        if target is None and (key.startswith("http") or "." in key and " " not in key):
            target = key if key.startswith("http") else "https://" + key
        before = {w.NativeWindowHandle for w in self._top_windows()}
        if target:
            try:
                os.startfile(target)
                title = self._activate_new_window(before, key.split(".")[0])
                return f"opened {name}" + (f" ({title})" if title else "")
            except OSError:
                pass
        # fall back to Windows search: works for any installed app
        self.kb.press(Key.cmd)
        self.kb.release(Key.cmd)
        time.sleep(0.6)
        self.kb.type(name)
        time.sleep(0.9)
        self.kb.press(Key.enter)
        self.kb.release(Key.enter)
        title = self._activate_new_window(before, key)
        return f"launched {name} via search" + (f" ({title})" if title else "")

    def do_click(self, target: str = "", **_) -> str:
        snap = scr.snapshot()
        self.last_snapshot = snap
        el = None
        if target.strip().isdigit():
            n = int(target)
            el = next((e for e in snap.elements if e.id == n), None)
        if el is None:
            el = snap.find(target)
        if el is None:
            return f"could not find '{target}' on screen"
        how = scr.invoke(el)
        return f"clicked {el.short_kind()} '{el.name}' ({how})"

    def do_type_text(self, text: str = "", **_) -> str:
        self.kb.type(text)
        return f"typed {len(text)} chars"

    def do_press_keys(self, keys: str = "", **_) -> str:
        parts = [p.strip().lower() for p in keys.replace(" plus ", "+").split("+") if p.strip()]
        mods = [KEY_MAP[p] for p in parts[:-1] if p in KEY_MAP]
        last = parts[-1] if parts else ""
        k = KEY_MAP.get(last, last if len(last) == 1 else None)
        if k is None:
            return f"unknown key {last}"
        for m in mods:
            self.kb.press(m)
        self.kb.press(k)
        self.kb.release(k)
        for m in reversed(mods):
            self.kb.release(m)
        return f"pressed {keys}"

    def do_scroll(self, direction: str = "down", amount: int = 3, **_) -> str:
        n = int(amount) if str(amount).lstrip("-").isdigit() else 3
        win32.scroll(-n if direction == "down" else n)
        return f"scrolled {direction} {n}"

    def do_focus_window(self, title: str = "", **_) -> str:
        import uiautomation as auto

        q = title.strip().lower()
        root = auto.GetRootControl()
        for w in root.GetChildren():
            if q and q in (w.Name or "").lower():
                try:
                    w.SetActive()
                    w.SetFocus()
                except Exception:
                    pass
                return f"focused '{w.Name}'"
        return f"no window matching '{title}'"

    def do_read_screen(self, **_) -> str:
        snap = scr.snapshot()
        self.last_snapshot = snap
        text = snap.as_text(max_items=60)
        spoken = self.summarise(text) if self.summarise else self._plain_summary(snap)
        self.say(spoken)
        return spoken

    @staticmethod
    def _plain_summary(snap: scr.Snapshot) -> str:
        kinds: dict[str, list[str]] = {}
        for e in snap.elements:
            if e.name:
                kinds.setdefault(e.short_kind(), []).append(e.name)
        parts = [f"You are in {snap.window_title or 'an unnamed window'}."]
        for k in ("Button", "Edit", "Hyperlink", "MenuItem", "TabItem", "ListItem", "Text"):
            if k in kinds:
                names = kinds[k][:6]
                parts.append(f"{len(kinds[k])} {k.lower()}s, including {', '.join(names)}.")
        return " ".join(parts)

    def do_read_focused(self, **_) -> str:
        text = scr.focused_text(1500) or "Nothing is selected."
        self.say(text)
        return text

    def do_describe_screen(self, **_) -> str:
        if not self.describe_screen:
            return "vision model unavailable"
        text = self.describe_screen()
        self.say(text)
        return text

    def do_describe_camera(self, **_) -> str:
        if not self.describe_camera:
            return "camera description unavailable"
        text = self.describe_camera()
        self.say(text)
        return text

    def do_cursor(self, action: str = "", **_) -> str:
        if action in ("left_click", "right_click", "double_click"):
            win32.click("right" if action == "right_click" else "left", 2 if action == "double_click" else 1)
            return action
        if self.cursor_action:
            self.cursor_action(action)
        return f"cursor {action}"

    def do_say(self, text: str = "", **_) -> str:
        self.say(text)
        return text

    def do_stop(self, **_) -> str:
        if self.cursor_action:
            self.cursor_action("stop_speaking")
        return "stopped"
