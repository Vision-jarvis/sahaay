"""Paths and user settings."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_NAME = "Sahaay"
REPO_ROOT = Path(__file__).resolve().parents[1]
USER_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / APP_NAME
USER_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = USER_DIR / "settings.json"


def models_root() -> Path:
    """Where model folders live. Env var wins; then a user install; then the dev checkout layout."""
    env = os.environ.get("SAHAAY_MODELS")
    if env:
        return Path(env)
    user = USER_DIR / "models"
    if (user / "mediapipe_face").exists():
        return user
    return REPO_ROOT.parent / "models" / "aihub"


def model_dir(kind: str) -> Path:
    root = models_root()
    candidates = {
        "face": [root / "mediapipe_face", root / "mediapipe_face_x2" / "mediapipe_face-onnx-float"],
        "whisper": [root / "whisper_base", *sorted(root.glob("whisper_base_x2/whisper_base-precompiled_qnn_onnx-*"))],
        "piper": [root / "piper", REPO_ROOT.parent / "models" / "piper"],
    }[kind]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


@dataclass
class Settings:
    head_cursor: bool = True
    voice: bool = True
    narrator_tts: bool = True
    language: str = "en"          # whisper language hint: en | hi
    dead_zone: float = 0.06
    gain: float = 1800.0
    smoothing: float = 0.35
    dwell_enabled: bool = True
    dwell_time: float = 0.9
    blink_enabled: bool = True
    ear_threshold: float = 0.17
    mouth_threshold: float = 0.42
    mic_threshold: float = 0.012
    llm_model: str = "ai-hub-models/Qwen3-4B-Instruct-2507"
    vlm_model: str = "ai-hub-models/Qwen3-VL-4B-Instruct"
    load_vlm: bool = True
    hud: bool = True
    bridge_port: str = ""          # COM port of an Arduino running arduino/sahaay_switch; empty = auto-detect
    hotkeys: dict = field(default_factory=lambda: {"toggle_head": "<ctrl>+<alt>+h", "toggle_voice": "<ctrl>+<alt>+v",
                                                   "calibrate": "<ctrl>+<alt>+c", "quit": "<ctrl>+<alt>+q"})

    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        if SETTINGS_PATH.exists():
            try:
                data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                for k, v in data.items():
                    if hasattr(s, k):
                        setattr(s, k, v)
            except Exception:
                pass
        return s

    def save(self) -> None:
        SETTINGS_PATH.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
