"""Sahaay application: wires the NPU models to the camera, microphone, screen and speaker.

Threads
-------
- camera loop: face tracker (NPU) -> head cursor (CPU), ~30 fps
- voice loop: mic VAD (CPU) -> Whisper (NPU) -> brain (grammar CPU / Qwen3 NPU) -> executor (CPU)
- model loaders: Qwen3-4B and Qwen3-VL-4B load in the background so the cursor works within seconds
- TTS worker inside Speaker
- Tk main thread: HUD overlay and dwell ring
"""
from __future__ import annotations

import threading
import time
import traceback

import numpy as np

from . import screen as scr
from .brain import Brain, ToolCall, summarise_screen
from .config import Settings, model_dir
from .control.actions import Executor
from .control.cursor import CursorConfig, HeadCursor
from .speech.mic import Microphone
from .speech.whisper import WhisperNPU
from .tts import Speaker
from .ui.hud import Hud, HudState
from .vision.camera import Camera
from .vision.describe import Describer
from .vision.face import FaceTracker


class Sahaay:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or Settings.load()
        self.hud_state = HudState(head_on=self.s.head_cursor, voice_on=self.s.voice)
        self.stop_event = threading.Event()
        self.latest_frame: np.ndarray | None = None
        self.tracker: FaceTracker | None = None
        self.cursor = HeadCursor(CursorConfig(dead_zone=self.s.dead_zone, gain=self.s.gain, smoothing=self.s.smoothing,
                                              dwell_enabled=self.s.dwell_enabled, dwell_time=self.s.dwell_time,
                                              blink_enabled=self.s.blink_enabled, ear_threshold=self.s.ear_threshold,
                                              mouth_threshold=self.s.mouth_threshold))
        self.cursor.st.enabled = self.s.head_cursor
        self.speaker: Speaker | None = None
        self.asr: WhisperNPU | None = None
        self.brain: Brain | None = None
        self.describer: Describer | None = None
        self.mic = Microphone(threshold=self.s.mic_threshold)
        self.executor = Executor(say=self.say, describe_screen=self._describe_screen, describe_camera=self._describe_camera,
                                 cursor_action=self._cursor_action, summarise=self._summarise)
        self._calibrate_request = threading.Event()
        self.hud: Hud | None = None

    # ---- speech out ------------------------------------------------------------------------
    def say(self, text: str) -> None:
        if not text:
            return
        self.hud_state.action = text[:160]
        if self.speaker and self.s.narrator_tts:
            self.speaker.say(text, interrupt=False)
        else:
            print("[say]", text)

    # ---- callbacks for the executor ------------------------------------------------------
    def _describe_screen(self) -> str:
        if not self.describer:
            return "The vision model is disabled in settings."
        self.hud_state.status = "thinking"
        try:
            self.hud_state.note = "VLM: screen"
            out = self.describer.describe_screen()
            self.hud_state.latencies["vlm"] = (self.describer.last_ms, "NPU")
            return out
        finally:
            self.hud_state.note = ""

    def _describe_camera(self) -> str:
        if not self.describer:
            return "The vision model is disabled in settings."
        if self.latest_frame is None:
            return "The camera is not running."
        self.hud_state.status = "thinking"
        out = self.describer.describe_camera(self.latest_frame)
        self.hud_state.latencies["vlm"] = (self.describer.last_ms, "NPU")
        return out

    def _summarise(self, screen_text: str) -> str:
        if self.brain and self.brain.ready.is_set():
            out = summarise_screen(self.brain, screen_text)
            self.hud_state.latencies["llm"] = (self.brain.last_ms, "NPU")
            if out:
                return out
        return Executor._plain_summary(self.executor.last_snapshot) if self.executor.last_snapshot else "Nothing to read."

    def _cursor_action(self, action: str) -> None:
        if action == "pause":
            self.cursor.st.enabled = False
            self.hud_state.head_on = False
        elif action == "resume":
            self.cursor.st.enabled = True
            self.hud_state.head_on = True
        elif action == "calibrate":
            self._calibrate_request.set()
        elif action == "stop_speaking" and self.speaker:
            self.speaker.stop()

    # ---- loops -----------------------------------------------------------------------------
    def camera_loop(self) -> None:
        try:
            self.tracker = FaceTracker(model_dir("face"))
            det, lmk = self.tracker.infos()
            unit = "NPU" if self.tracker.on_npu else "CPU"
            cam = Camera(width=1280, height=720)
            cam.start()
        except Exception as exc:
            self.hud_state.note = f"camera/face error: {exc}"
            traceback.print_exc()
            return
        calib: list[tuple[float, float]] = []
        calib_until = time.time() + 1.2  # auto-calibrate at start
        lat_t = time.time()
        try:
            while not self.stop_event.is_set():
                frame = cam.read(timeout=0.5)
                if frame is None:
                    continue
                self.latest_frame = frame
                r = self.tracker.process(frame)
                self.hud_state.face_found = r.found
                if self._calibrate_request.is_set():
                    self._calibrate_request.clear()
                    calib = []
                    calib_until = time.time() + 1.2
                    self.say("Hold still, calibrating.")
                if r.found and time.time() < calib_until:
                    calib.append((r.yaw, r.pitch))
                    continue
                if calib:
                    ys, ps = zip(*calib)
                    self.cursor.calibrate(float(np.median(ys)), float(np.median(ps)))
                    calib = []
                    self.hud_state.note = "calibrated"
                ev = self.cursor.update(r)
                self.hud_state.dwell = self.cursor.dwell_progress
                if ev:
                    self.hud_state.action = ev.replace(":", " ").replace("_", " ")
                if time.time() - lat_t > 0.5:
                    lat_t = time.time()
                    self.hud_state.latencies["face"] = (r.detector_ms + r.landmark_ms, unit)
        finally:
            cam.stop()

    def voice_loop(self) -> None:
        import comtypes

        try:
            comtypes.CoInitialize()  # UI Automation and DirectShow are COM
        except OSError:
            pass
        try:
            self.asr = WhisperNPU(model_dir("whisper"))
            unit = "NPU" if self.asr.on_npu else "CPU"
            self.mic.start()
            self.mic.calibrate_noise(0.8)
        except Exception as exc:
            self.hud_state.note = f"mic/asr error: {exc}"
            traceback.print_exc()
            return
        self.hud_state.status = "listening"
        while not self.stop_event.is_set():
            self.mic.muted = bool(self.speaker and self.speaker.speaking) or not self.s.voice
            self.hud_state.mic_level = self.mic.level
            if self.mic.speaking and self.hud_state.status == "listening":
                self.hud_state.status = "hearing"
            try:
                audio = self.mic.utterances.get(timeout=0.1)
            except Exception:
                if self.hud_state.status == "hearing" and not self.mic.speaking:
                    self.hud_state.status = "listening"
                continue
            if not self.s.voice:
                continue
            self.hud_state.status = "thinking"
            try:
                t = self.asr.transcribe(audio, language=self.s.language or None)
                self.hud_state.latencies["whisper"] = (t.encoder_ms + t.decoder_ms, unit)
                text = t.text.strip()
                if len(text) < 2:
                    self.hud_state.status = "listening"
                    continue
                self.hud_state.heard = text
                self.handle_utterance(text)
            except Exception as exc:
                self.hud_state.note = f"voice error: {exc}"
                traceback.print_exc()
            self.hud_state.status = "listening"

    def handle_utterance(self, text: str) -> list[str]:
        calls: list[ToolCall]
        needs_screen = not any(text.lower().startswith(v) for v in ("type ", "press ", "scroll", "open ", "sahaay"))
        screen_text = ""
        if self.brain:
            if needs_screen:
                try:
                    snap = scr.snapshot()
                    self.executor.last_snapshot = snap
                    screen_text = snap.as_text(max_items=80)
                except Exception:
                    screen_text = ""
            calls = self.brain.plan(text, screen_text)
            if calls and calls[0].source == "llm":
                self.hud_state.latencies["llm"] = (self.brain.last_ms, "NPU")
        else:
            from .brain import fast_parse

            calls = fast_parse(text) or [ToolCall("say", {"text": "I heard you, but my language model is not loaded yet."})]
        self.hud_state.action = "; ".join(str(c) for c in calls)[:160]
        results = self.executor.run(calls)
        print(f"[heard] {text!r} -> {[str(c) for c in calls]} -> {results}")
        return results

    # ---- lifecycle -------------------------------------------------------------------------
    def start(self) -> None:
        try:
            self.speaker = Speaker(model_dir("piper"))
        except Exception as exc:
            print("TTS unavailable:", exc)
        # GenieX bundles must be loaded one at a time (concurrent loads fail to create HTP contexts),
        # so the LLM loads first and the VLM follows on the same thread.
        self.brain = Brain(self.s.llm_model, preload=False)
        if self.s.load_vlm:
            self.describer = Describer(self.s.vlm_model, preload=False)

        def load_models() -> None:
            self.brain._load()
            if self.describer:
                self.describer._load()

        threading.Thread(target=self.camera_loop, daemon=True, name="camera").start()
        threading.Thread(target=self.voice_loop, daemon=True, name="voice").start()
        threading.Thread(target=load_models, daemon=True, name="models").start()
        threading.Thread(target=self._announce_ready, daemon=True).start()
        self._start_hotkeys()

    def _announce_ready(self) -> None:
        if self.brain:
            self.brain.ready.wait(180)
            self.hud_state.latencies["llm"] = (self.brain.load_ms, "NPU load")
        self.say("Sahaay is ready.")

    def _start_hotkeys(self) -> None:
        try:
            from pynput import keyboard

            hk = self.s.hotkeys
            keyboard.GlobalHotKeys({
                hk["toggle_head"]: lambda: self._cursor_action("pause" if self.cursor.st.enabled else "resume"),
                hk["toggle_voice"]: self._toggle_voice,
                hk["calibrate"]: lambda: self._cursor_action("calibrate"),
                hk["quit"]: self.quit,
            }).start()
        except Exception as exc:
            print("hotkeys unavailable:", exc)

    def _toggle_voice(self) -> None:
        self.s.voice = not self.s.voice
        self.hud_state.voice_on = self.s.voice

    def run(self) -> None:
        self.start()
        self.hud = Hud(self.hud_state, on_quit=self.quit)
        self.hud.set_visible(self.s.hud)
        import os

        auto_quit = float(os.environ.get("SAHAAY_AUTOQUIT_S", "0") or 0)
        if auto_quit > 0:  # used by automated tests
            self.hud.root.after(int(auto_quit * 1000), self.quit)
        try:
            from .ui.tray import start_tray

            start_tray(self)
        except Exception as exc:
            print("tray unavailable:", exc)
        self.hud.run()
        self.quit()

    def quit(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        try:
            if self.cursor.st.dragging:
                from .control import win32

                win32.mouse_up("left")
            self.mic.stop()
            if self.brain:
                self.brain.close()
            if self.describer:
                self.describer.close()
        except Exception:
            pass
        if self.hud:
            self.hud.quit()
