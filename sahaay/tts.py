"""Text to speech: Piper VITS voices (English and Hindi) with an espeak-ng phonemizer.

The ``piper-tts`` wheel ships a compiled espeak bridge with no ARM64 build, so this module
phonemizes through the espeak-ng CLI (installed via winget, runs under x64 emulation in a
few ms) and runs the Piper ONNX voice with ONNX Runtime directly. The voice graph is tried on
the NPU first and falls back to CPU if QNN cannot compile its dynamic shapes; either way it is
real-time. Playback goes through sounddevice on a worker thread so callers never block.
"""
from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .npu import create_session

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
ESPEAK_CANDIDATES = [
    r"C:\Program Files\eSpeak NG\espeak-ng.exe",
    r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
    "espeak-ng",
]


def find_espeak() -> str | None:
    import shutil

    for c in ESPEAK_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    return None


@dataclass
class Voice:
    name: str
    session: ort.InferenceSession
    on_npu: bool
    sample_rate: int
    phoneme_ids: dict[str, list[int]]
    espeak_voice: str
    noise_scale: float
    length_scale: float
    noise_w: float
    speaker_id: int | None


class Speaker:
    def __init__(self, voices_dir: str | Path, en: str = "en_US-lessac-medium", hi: str = "hi_IN-pratham-medium",
                 prefer_npu: bool = False) -> None:
        voices_dir = Path(voices_dir)
        self.espeak = find_espeak()
        if not self.espeak:
            raise FileNotFoundError("espeak-ng not found; install with: winget install eSpeak-NG.eSpeak-NG")
        self.voices: dict[str, Voice] = {}
        for lang, name in (("en", en), ("hi", hi)):
            p = voices_dir / f"{name}.onnx"
            if p.exists():
                self.voices[lang] = self._load(name, p, prefer_npu)
        if not self.voices:
            raise FileNotFoundError(f"no Piper voices in {voices_dir}")
        self._q: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self.speaking = False
        self.last_synth_ms = 0.0
        threading.Thread(target=self._worker, daemon=True).start()

    @staticmethod
    def _load(name: str, onnx_path: Path, prefer_npu: bool) -> Voice:
        cfg = json.loads(Path(str(onnx_path) + ".json").read_text(encoding="utf-8"))
        on_npu = False
        sess = None
        if prefer_npu:
            try:
                sess, info = create_session(onnx_path, prefer_npu=True, context_cache=False)
                on_npu = info.on_npu
            except Exception:
                sess = None
        if sess is None:
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inf = cfg.get("inference", {})
        return Voice(name, sess, on_npu, int(cfg["audio"]["sample_rate"]), cfg["phoneme_id_map"],
                     cfg.get("espeak", {}).get("voice", "en-us"), float(inf.get("noise_scale", 0.667)),
                     float(inf.get("length_scale", 1.0)), float(inf.get("noise_w", 0.8)),
                     0 if int(cfg.get("num_speakers", 1)) > 1 else None)

    # ---- phonemes -------------------------------------------------------------------------
    def phonemize(self, text: str, voice: Voice) -> list[str]:
        """Returns IPA phoneme strings, one per sentence (espeak splits on terminators)."""
        out = subprocess.run([self.espeak, "-q", "--ipa", "-v", voice.espeak_voice, "--stdin"],
                             input=text.encode("utf-8"), capture_output=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        ipa = out.stdout.decode("utf-8", "replace").replace("\r", "")
        return [s.strip() for s in ipa.split("\n") if s.strip()]

    @staticmethod
    def to_ids(phonemes: str, voice: Voice) -> np.ndarray:
        m = voice.phoneme_ids
        ids: list[int] = list(m.get("^", [1]))
        pad = m.get("_", [0])
        for ch in phonemes:
            if ch in m:
                ids.extend(m[ch])
                ids.extend(pad)
        ids.extend(m.get("$", [2]))
        return np.array([ids], dtype=np.int64)

    # ---- synthesis ------------------------------------------------------------------------
    def pick_lang(self, text: str) -> str:
        if DEVANAGARI.search(text) and "hi" in self.voices:
            return "hi"
        return "en" if "en" in self.voices else next(iter(self.voices))

    def synthesize(self, text: str, lang: str | None = None) -> tuple[np.ndarray, int]:
        voice = self.voices[lang or self.pick_lang(text)]
        t0 = time.perf_counter()
        chunks = []
        for sent in self.phonemize(text, voice):
            ids = self.to_ids(sent, voice)
            feeds = {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                     "scales": np.array([voice.noise_scale, voice.length_scale, voice.noise_w], dtype=np.float32)}
            if voice.speaker_id is not None:
                feeds["sid"] = np.array([voice.speaker_id], dtype=np.int64)
            audio = voice.session.run(None, feeds)[0].reshape(-1).astype(np.float32)
            chunks.append(audio)
            chunks.append(np.zeros(int(0.12 * voice.sample_rate), np.float32))  # sentence gap
        self.last_synth_ms = (time.perf_counter() - t0) * 1000
        return (np.concatenate(chunks) if chunks else np.zeros(0, np.float32)), voice.sample_rate

    def say(self, text: str, lang: str | None = None, interrupt: bool = False) -> None:
        if interrupt:
            self.stop()
        self._q.put((text, lang or self.pick_lang(text)))

    def stop(self) -> None:
        with self._q.mutex:
            self._q.queue.clear()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    def _worker(self) -> None:
        import sounddevice as sd

        while True:
            item = self._q.get()
            if item is None:
                break
            text, lang = item
            try:
                audio, rate = self.synthesize(text, lang)
                if audio.size:
                    self.speaking = True
                    sd.play(audio, rate)
                    sd.wait()
            except Exception as exc:
                print("tts error:", exc)
            finally:
                self.speaking = False
