"""Text to speech with Piper voices (English and Hindi), played through sounddevice.

Piper runs its VITS decoder with ONNX Runtime. On Snapdragon it currently runs on CPU inside
the piper package; the standalone Piper ONNX from Qualcomm AI Hub is the NPU path and slots in
here without changing callers.
"""
from __future__ import annotations

import queue
import re
import threading
from pathlib import Path

import numpy as np

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


class Speaker:
    def __init__(self, voices_dir: str | Path, en: str = "en_US-lessac-medium", hi: str = "hi_IN-pratham-medium") -> None:
        from piper import PiperVoice

        voices_dir = Path(voices_dir)
        self.voices = {}
        for lang, name in (("en", en), ("hi", hi)):
            p = voices_dir / f"{name}.onnx"
            if p.exists():
                self.voices[lang] = PiperVoice.load(str(p))
        if not self.voices:
            raise FileNotFoundError(f"no Piper voices in {voices_dir}")
        self._q: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self.speaking = False

    def pick_lang(self, text: str) -> str:
        return "hi" if DEVANAGARI.search(text) and "hi" in self.voices else ("en" if "en" in self.voices else next(iter(self.voices)))

    def synthesize(self, text: str, lang: str | None = None) -> tuple[np.ndarray, int]:
        voice = self.voices[lang or self.pick_lang(text)]
        chunks = []
        rate = 22050
        if hasattr(voice, "synthesize"):
            for ch in voice.synthesize(text):  # piper >= 1.3: AudioChunk objects
                arr = getattr(ch, "audio_float_array", None)
                if arr is None:
                    arr = np.frombuffer(ch.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                rate = getattr(ch, "sample_rate", rate)
                chunks.append(arr)
        else:  # older API
            for b in voice.synthesize_stream_raw(text):
                chunks.append(np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0)
            rate = voice.config.sample_rate
        return (np.concatenate(chunks) if chunks else np.zeros(0, np.float32)), rate

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

        while not self._stop.is_set():
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
            except Exception as exc:  # keep the worker alive
                print("tts error:", exc)
            finally:
                self.speaking = False
