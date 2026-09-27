"""The brain: turns what the user said into tool calls.

Two tiers, so the app feels instant:
1. A fast grammar of common commands (regex, sub-millisecond, CPU).
2. Qwen3-4B on the Hexagon NPU through GenieX for anything free-form, with a strict JSON
   tool schema and a compact UI Automation snapshot of the screen as context.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

TOOLS: dict[str, str] = {
    "open_app": "open an application by name (notepad, calculator, chrome, edge, explorer, settings, word, ...)",
    "click": "click a UI element by its visible name or number from the screen list",
    "type_text": "type text into the focused field",
    "press_keys": "press a key combo, e.g. 'enter', 'ctrl+s', 'alt+tab', 'win+d'",
    "scroll": "scroll 'up' or 'down' by an amount (1-10)",
    "focus_window": "bring a window to front by (part of) its title",
    "read_screen": "read a summary of what is on screen",
    "read_focused": "read the text of the focused element or paragraph",
    "describe_screen": "describe the screen visually (images, charts, photos) with the vision model",
    "describe_camera": "describe what the webcam sees",
    "cursor": "control the head cursor: 'pause', 'resume', 'calibrate', 'left_click', 'right_click', 'double_click'",
    "say": "speak a short reply to the user (for questions or confirmations)",
    "stop": "stop speaking / cancel",
}

TOOL_ARGS: dict[str, tuple[str, ...]] = {
    "open_app": ("name",), "click": ("target",), "type_text": ("text",), "press_keys": ("keys",),
    "scroll": ("direction", "amount"), "focus_window": ("title",), "read_screen": (), "read_focused": (),
    "describe_screen": (), "describe_camera": (), "cursor": ("action",), "say": ("text",), "stop": (),
}
_ARG_ALIASES = {"name": "target", "element": "target", "number": "target", "id": "target", "label": "target",
                "button": "target", "app": "name", "application": "name", "message": "text", "content": "text",
                "key": "keys", "combo": "keys", "window": "title", "dir": "direction"}

SYSTEM_PROMPT = (
    "You are Sahaay, an offline accessibility assistant that operates a Windows PC for people who cannot use a mouse, "
    "keyboard or screen. Convert the user's request into a JSON list of tool calls.\nTools (with their exact argument names):\n"
    + "\n".join(f"- {k}({', '.join(TOOL_ARGS[k])}): {v}" for k, v in TOOLS.items())
    + "\nReply with ONLY a JSON list. Example:\n"
    '[{"tool": "open_app", "args": {"name": "notepad"}}, {"tool": "type_text", "args": {"text": "Hi Riya, hope you are well!"}}, '
    '{"tool": "press_keys", "args": {"keys": "ctrl+s"}}]\n'
    "Rules: when asked to write or compose text, put the full composed text in type_text. For click, target is the element's "
    "visible name or its number from the screen list; never invent elements. For questions about the screen, answer with say. "
    "At most 5 calls."
)


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    source: str = "llm"

    def __str__(self) -> str:
        return f"{self.tool}({', '.join(f'{k}={v!r}' for k, v in self.args.items())})"


# ---- tier 1: fast grammar --------------------------------------------------------------------
_FAST: list[tuple[re.Pattern[str], Any]] = []


def _rule(pattern: str, make):
    _FAST.append((re.compile(pattern, re.IGNORECASE), make))


_rule(r"^(sahaay[, ]+)?(stop|cancel|quiet|chup)\.?$", lambda m: [ToolCall("stop")])
_rule(r"^(sahaay[, ]+)?(pause|sleep|go to sleep|rest)( cursor| tracking)?\.?$", lambda m: [ToolCall("cursor", {"action": "pause"})])
_rule(r"^(sahaay[, ]+)?(resume|wake( up)?|continue)( cursor| tracking)?\.?$", lambda m: [ToolCall("cursor", {"action": "resume"})])
_rule(r"^(sahaay[, ]+)?(re)?calibrate\.?$", lambda m: [ToolCall("cursor", {"action": "calibrate"})])
_rule(r"^(sahaay[, ]+)?(left )?click\.?$", lambda m: [ToolCall("cursor", {"action": "left_click"})])
_rule(r"^(sahaay[, ]+)?right[- ]?click\.?$", lambda m: [ToolCall("cursor", {"action": "right_click"})])
_rule(r"^(sahaay[, ]+)?double[- ]?click\.?$", lambda m: [ToolCall("cursor", {"action": "double_click"})])
_rule(r"^(sahaay[, ]+)?click (on )?(the )?(?P<t>.+?)( button| link| tab)?\.?$", lambda m: [ToolCall("click", {"target": m.group("t")})])
_rule(r"^(sahaay[, ]+)?(press |hit )(?P<k>enter|return|escape|esc|tab|space|backspace|delete|home|end|page ?up|page ?down|up|down|left|right)\.?$",
      lambda m: [ToolCall("press_keys", {"keys": m.group("k").replace(" ", "").replace("return", "enter").replace("esc", "escape") if m.group("k") != "escape" else "escape"})])
_rule(r"^(sahaay[, ]+)?(press |hit )(?P<k>(ctrl|control|alt|shift|win|windows)( ?\+ ?| plus | )[a-z0-9]+)\.?$",
      lambda m: [ToolCall("press_keys", {"keys": re.sub(r"\s*(\+|plus)\s*|\s+", "+", m.group("k").lower()).replace("control", "ctrl").replace("windows", "win")})])
_rule(r"^(sahaay[, ]+)?scroll (?P<d>up|down)( (?P<n>\d+|a lot|a little))?\.?$",
      lambda m: [ToolCall("scroll", {"direction": m.group("d"), "amount": 8 if m.group("n") == "a lot" else 1 if m.group("n") == "a little" else int(m.group("n") or 3)})])
_rule(r"^(sahaay[, ]+)?(open|launch|start) (?P<a>.+?)\.?$", lambda m: [ToolCall("open_app", {"name": m.group("a")})])
_rule(r"^(sahaay[, ]+)?(switch to|go to|focus) (?P<w>.+?)( window)?\.?$", lambda m: [ToolCall("focus_window", {"title": m.group("w")})])
_rule(r"^(sahaay[, ]+)?(next window|switch window)\.?$", lambda m: [ToolCall("press_keys", {"keys": "alt+tab"})])
_rule(r"^(sahaay[, ]+)?(type|likho) (?P<t>.+)$", lambda m: [ToolCall("type_text", {"text": m.group("t")})])
_rule(r"^(sahaay[, ]+)?(what('s| is) on (my |the )?screen|read (the |my )?screen|screen summary)\??\.?$", lambda m: [ToolCall("read_screen")])
_rule(r"^(sahaay[, ]+)?(read (this|it|that|the (focused|selected|current) (text|paragraph|element))|read)\.?$", lambda m: [ToolCall("read_focused")])
_rule(r"^(sahaay[, ]+)?(describe|explain) (the |my )?(screen|image|picture|chart|photo|this)\.?$", lambda m: [ToolCall("describe_screen")])
_rule(r"^(sahaay[, ]+)?(describe|what do you see|what('s| is) in front of me|look) ?(the |through the |at the )?(camera|webcam|room|around)?\.?$", lambda m: [ToolCall("describe_camera")])
_rule(r"^(sahaay[, ]+)?(select all)\.?$", lambda m: [ToolCall("press_keys", {"keys": "ctrl+a"})])
_rule(r"^(sahaay[, ]+)?(copy|paste|undo|save|cut)( (this|that|it))?\.?$",
      lambda m: [ToolCall("press_keys", {"keys": {"copy": "ctrl+c", "paste": "ctrl+v", "undo": "ctrl+z", "save": "ctrl+s", "cut": "ctrl+x"}[m.group(2)]})])
_rule(r"^(sahaay[, ]+)?(close (this|the) window|close window)\.?$", lambda m: [ToolCall("press_keys", {"keys": "alt+f4"})])
_rule(r"^(sahaay[, ]+)?(show (the )?desktop|minimize everything)\.?$", lambda m: [ToolCall("press_keys", {"keys": "win+d"})])


def fast_parse(text: str) -> list[ToolCall] | None:
    t = text.strip().strip(".!?").strip()
    for pat, make in _FAST:
        m = pat.match(t)
        if m:
            calls = make(m)
            for c in calls:
                c.source = "grammar"
            return calls
    return None


# ---- tier 2: SLM on the NPU ------------------------------------------------------------------
class Brain:
    def __init__(self, model_id: str = "ai-hub-models/Qwen3-4B-Instruct-2507", preload: bool = True) -> None:
        self.model_id = model_id
        self._llm = None
        self._lock = threading.Lock()
        self.ready = threading.Event()
        self.load_ms = 0.0
        self.last_ms = 0.0
        self.last_tokens = 0
        if preload:
            threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        from geniex import AutoModelForCausalLM

        t0 = time.perf_counter()
        llm = AutoModelForCausalLM.from_pretrained(self.model_id)
        with self._lock:
            self._llm = llm
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.ready.set()

    def plan(self, utterance: str, screen_text: str = "", timeout: float = 30.0) -> list[ToolCall]:
        fast = fast_parse(utterance)
        if fast:
            return fast
        if not self.ready.wait(timeout):
            return [ToolCall("say", {"text": "I am still loading my language model, one moment."}, source="fallback")]
        user = utterance.strip()
        if screen_text:
            user = f"Screen:\n{screen_text}\n\nUser said: {utterance.strip()}"
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        with self._lock:
            llm = self._llm
            prompt = llm.tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
            t0 = time.perf_counter()
            out = llm.generate(prompt, max_new_tokens=200, temperature=0.0, stop=["]\n", "\n\n"])
            self.last_ms = (time.perf_counter() - t0) * 1000
        text = out if isinstance(out, str) else str(out)
        calls = parse_tool_calls(text)
        if not calls:
            calls = [ToolCall("say", {"text": text.strip()[:200] or "Sorry, I did not understand that."}, source="fallback")]
        return calls

    def close(self) -> None:
        with self._lock:
            if self._llm is not None:
                self._llm.close()
                self._llm = None


def parse_tool_calls(text: str) -> list[ToolCall]:
    s = text.strip()
    s = re.sub(r"^```(json)?|```$", "", s, flags=re.MULTILINE).strip()
    start, end = s.find("["), s.rfind("]")
    if start < 0:
        return []
    if end < start:
        s = s[start:] + "]"
    else:
        s = s[start : end + 1]
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        try:
            data = json.loads(s.rstrip(", \n") + "]") if not s.endswith("]") else []
        except json.JSONDecodeError:
            return []
    calls = []
    for item in data if isinstance(data, list) else [data]:
        if not isinstance(item, dict):
            continue
        tool = item.get("tool") or item.get("action") or item.get("name")
        if tool not in TOOLS:
            continue
        args = item.get("args") or item.get("arguments") or item.get("parameters") or {k: v for k, v in item.items() if k not in ("tool", "action", "name")}
        args = dict(args) if isinstance(args, dict) else {}
        allowed = TOOL_ARGS.get(str(tool), ())
        norm: dict[str, Any] = {}
        for k, v in args.items():
            k2 = k if k in allowed else _ARG_ALIASES.get(k, k)
            if k2 in allowed:
                norm[k2] = v if not isinstance(v, (int, float)) or k2 == "amount" else str(v)
        if str(tool) == "click" and "target" not in norm and args:
            norm["target"] = str(next(iter(args.values())))
        calls.append(ToolCall(str(tool), norm))
    return calls[:5]


SUMMARY_PROMPT = ("Summarise this screen for a blind user in two short spoken sentences: which app or page it is, what the "
                  "main content is, and the most useful buttons or fields by name. Plain text, no lists, no markdown.")


def summarise_screen(brain: "Brain", screen_text: str, timeout: float = 30.0) -> str:
    if not brain.ready.wait(timeout):
        return ""
    msgs = [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": screen_text[:6000]}]
    with brain._lock:
        llm = brain._llm
        prompt = llm.tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        t0 = time.perf_counter()
        out = llm.generate(prompt, max_new_tokens=90, temperature=0.0)
        brain.last_ms = (time.perf_counter() - t0) * 1000
    return (out if isinstance(out, str) else str(out)).strip()
