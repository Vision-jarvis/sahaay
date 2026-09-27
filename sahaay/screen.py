"""Screen understanding without a vision model: Windows UI Automation snapshot.

Walks the foreground window's accessibility tree and produces a compact, numbered list of
interactive and readable elements. This is what the on-device language model reasons over
(cheap, exact, and it works for every accessible app), and what click-by-name resolves against.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import uiautomation as auto

INTERACTIVE = {
    "ButtonControl", "CheckBoxControl", "ComboBoxControl", "EditControl", "HyperlinkControl",
    "ListItemControl", "MenuItemControl", "RadioButtonControl", "TabItemControl", "TreeItemControl",
    "SplitButtonControl", "SliderControl", "SpinnerControl", "DocumentControl",
}
READABLE = {"TextControl", "HeaderControl", "HeaderItemControl", "DataItemControl", "ImageControl", "StatusBarControl"}
SKIP = {"ScrollBarControl", "ThumbControl", "SeparatorControl", "TitleBarControl"}


@dataclass
class Element:
    id: int
    kind: str
    name: str
    value: str
    rect: tuple[int, int, int, int]  # left, top, right, bottom
    control: auto.Control

    @property
    def center(self) -> tuple[int, int]:
        l, t, r, b = self.rect
        return (l + r) // 2, (t + b) // 2

    def short_kind(self) -> str:
        return self.kind.replace("Control", "")


@dataclass
class Snapshot:
    window_title: str
    process: str
    elements: list[Element]
    took_ms: float

    def as_text(self, max_items: int = 120, max_len: int = 60) -> str:
        lines = [f'Window: "{self.window_title}" ({self.process})']
        for e in self.elements[:max_items]:
            val = f' = "{e.value[:max_len]}"' if e.value and e.value != e.name else ""
            lines.append(f'[{e.id}] {e.short_kind()} "{e.name[:max_len]}"{val}')
        if len(self.elements) > max_items:
            lines.append(f"... and {len(self.elements) - max_items} more")
        return "\n".join(lines)

    def find(self, query: str) -> Element | None:
        """Best-effort name match: exact, then prefix, then substring, case-insensitive."""
        q = query.strip().lower()
        if not q:
            return None
        for pred in (lambda n: n == q, lambda n: n.startswith(q), lambda n: q in n):
            for e in self.elements:
                if pred(e.name.lower()):
                    return e
        return None


def _value_of(c: auto.Control) -> str:
    try:
        vp = c.GetValuePattern()
        if vp:
            return str(vp.Value or "")
    except Exception:
        pass
    return ""


def snapshot(max_elements: int = 300, max_depth: int = 12, include_readable: bool = True) -> Snapshot:
    t0 = time.perf_counter()
    win = auto.GetForegroundControl()
    top = win.GetTopLevelControl() or win
    try:
        pname = top.ProcessId and auto.GetProcessCommandLine(top.ProcessId) or ""
    except Exception:
        pname = ""
    elements: list[Element] = []
    idx = 0
    for c, depth in auto.WalkControl(top, includeTop=False, maxDepth=max_depth):
        if len(elements) >= max_elements:
            break
        kind = c.ControlTypeName
        if kind in SKIP:
            continue
        try:
            if c.IsOffscreen:
                continue
            r = c.BoundingRectangle
            if r.width() <= 0 or r.height() <= 0:
                continue
        except Exception:
            continue
        name = (c.Name or "").strip()
        if kind in INTERACTIVE or (include_readable and kind in READABLE and name):
            if not name and kind not in {"EditControl", "DocumentControl"}:
                continue
            idx += 1
            elements.append(Element(idx, kind, name, _value_of(c) if kind in {"EditControl", "DocumentControl", "ComboBoxControl"} else "",
                                    (r.left, r.top, r.right, r.bottom), c))
    return Snapshot(top.Name or "", pname.split("\\")[-1].split(" ")[0] if pname else "", elements, (time.perf_counter() - t0) * 1000)


def invoke(e: Element) -> str:
    """Click an element the accessible way if possible, else by its centre point."""
    try:
        ip = e.control.GetInvokePattern()
        if ip:
            ip.Invoke()
            return "invoke"
    except Exception:
        pass
    try:
        if e.kind in {"CheckBoxControl", "RadioButtonControl"}:
            tp = e.control.GetTogglePattern()
            if tp:
                tp.Toggle()
                return "toggle"
        sp = e.control.GetSelectionItemPattern()
        if sp:
            sp.Select()
            return "select"
    except Exception:
        pass
    x, y = e.center
    e.control.Click(simulateMove=False)
    return f"click@{x},{y}"


def focused_text(max_len: int = 4000) -> str:
    """Text of the focused element (for 'read this')."""
    c = auto.GetFocusedControl()
    if c is None:
        return ""
    parts = []
    try:
        tp = c.GetTextPattern()
        if tp:
            parts.append(tp.DocumentRange.GetText(max_len))
    except Exception:
        pass
    if not parts:
        parts.append(_value_of(c) or c.Name or "")
    return " ".join(p for p in parts if p).strip()[:max_len]
