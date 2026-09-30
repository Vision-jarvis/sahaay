"""Microphone capture with a small energy-based voice activity detector.

Produces complete utterances (float32, 16 kHz) for Whisper. Runs on the CPU; the NPU only
sees finished utterances. ``muted`` is set while Sahaay itself is speaking so it does not
transcribe its own voice.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

SAMPLE_RATE = 16000
BLOCK = 480  # 30 ms


class Microphone:
    def __init__(self, *, threshold: float = 0.02, silence_ms: int = 700, min_ms: int = 500,
                 max_s: float = 15.0, pre_roll_ms: int = 300, device: int | None = None) -> None:
        self.threshold = threshold
        self.silence_blocks = max(1, silence_ms // 30)
        self.min_blocks = max(1, min_ms // 30)
        self.max_blocks = int(max_s * 1000 // 30)
        self.pre_roll = max(1, pre_roll_ms // 30)
        self.device = device
        self.utterances: queue.Queue[np.ndarray] = queue.Queue()
        self.muted = False
        self.level = 0.0
        self.speaking = False
        self._stream = None
        self._buf: list[np.ndarray] = []
        self._pre: list[np.ndarray] = []
        self._silence = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK,
                                      device=self.device, callback=self._cb)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def calibrate_noise(self, seconds: float = 1.0) -> float:
        """Sample ambient level and set the threshold a bit above it."""
        levels = []
        t_end = time.time() + seconds
        while time.time() < t_end:
            levels.append(self.level)
            time.sleep(0.03)
        noise = float(np.percentile(levels, 90)) if levels else 0.0
        self.threshold = max(0.02, noise * 3.0)
        return self.threshold

    def _cb(self, indata, frames, t, status) -> None:  # sounddevice thread
        block = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(block * block)) + 1e-9)
        self.level = rms
        if self.muted:
            self._buf.clear()
            self.speaking = False
            return
        voiced = rms > self.threshold
        with self._lock:
            if not self.speaking:
                self._pre.append(block)
                if len(self._pre) > self.pre_roll:
                    self._pre.pop(0)
                if voiced:
                    self.speaking = True
                    self._buf = list(self._pre)
                    self._silence = 0
            else:
                self._buf.append(block)
                self._silence = 0 if voiced else self._silence + 1
                if self._silence >= self.silence_blocks or len(self._buf) >= self.max_blocks:
                    voiced_blocks = len(self._buf) - self._silence
                    if voiced_blocks >= self.min_blocks:
                        self.utterances.put(np.concatenate(self._buf))
                    self._buf = []
                    self._pre = []
                    self.speaking = False
