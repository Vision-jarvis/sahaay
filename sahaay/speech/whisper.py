"""Whisper speech-to-text on the Hexagon NPU (Qualcomm AI Hub ``whisper_base``, precompiled QNN ONNX).

The AI Hub export splits Whisper into an encoder (mel -> cross-attention KV caches) and a
single-step decoder with explicit self-attention KV caches. This class re-implements the
reference greedy decode loop from qai_hub_models with numpy and ONNX Runtime so it runs on
ARM64 Windows without torch. Mel features and token decoding come from ``transformers``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..npu import SessionInfo, create_session

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
MEAN_DECODE_LEN = 200
MASK_NEG = -100.0  # qai_hub_models hf_whisper MASK_NEG: the graph was exported with this value

SOT, EOT = 50258, 50257
LANG = {"en": 50259, "hi": 50276}
TRANSCRIBE, NOTIMESTAMPS = 50359, 50363


@dataclass
class Transcript:
    text: str
    tokens: list[int]
    language: str
    encoder_ms: float
    decoder_ms: float
    steps: int
    audio_seconds: float


class WhisperNPU:
    def __init__(self, model_dir: str | Path, *, prefer_npu: bool = True, hf_id: str = "openai/whisper-base") -> None:
        from transformers import WhisperFeatureExtractor, WhisperTokenizer

        model_dir = Path(model_dir)
        self.enc, self.enc_info = create_session(model_dir / "encoder.onnx", prefer_npu=prefer_npu, context_cache=False)
        self.dec, self.dec_info = create_session(model_dir / "decoder.onnx", prefer_npu=prefer_npu, context_cache=False)
        self.fe = WhisperFeatureExtractor.from_pretrained(hf_id)
        self.tok = WhisperTokenizer.from_pretrained(hf_id)
        self.special = set(self.tok.all_special_ids)
        self.n_layers = sum(1 for i in self.dec.get_inputs() if i.name.startswith("k_cache_self_"))
        self.enc_out_names = [o.name for o in self.enc.get_outputs()]
        kin = next(i for i in self.dec.get_inputs() if i.name == "k_cache_self_0_in")
        self.heads, _, self.dim, self.cache_len = kin.shape
        self.dtype = np.float16 if "float16" in kin.type else np.float32

    @property
    def on_npu(self) -> bool:
        return self.enc_info.on_npu and self.dec_info.on_npu

    def infos(self) -> tuple[SessionInfo, SessionInfo]:
        return self.enc_info, self.dec_info

    # ---- public --------------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE, language: str | None = "en",
                   max_tokens: int = MEAN_DECODE_LEN - 1) -> Transcript:
        """audio: float32 mono in [-1, 1]. Longer than 30 s is truncated (callers chunk)."""
        if sample_rate != SAMPLE_RATE:
            from scipy.signal import resample_poly

            g = np.gcd(sample_rate, SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // g, sample_rate // g).astype(np.float32)
        audio = np.asarray(audio, dtype=np.float32)[:N_SAMPLES]
        secs = len(audio) / SAMPLE_RATE
        feats = self.fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")["input_features"].astype(self.dtype)

        t0 = time.perf_counter()
        cross = dict(zip(self.enc_out_names, self.enc.run(None, {"input_features": feats})))
        t1 = time.perf_counter()

        prefix = [SOT]
        if language in LANG:
            prefix += [LANG[language], TRANSCRIBE, NOTIMESTAMPS]
        tokens = list(prefix)
        k_self = [np.zeros((self.heads, 1, self.dim, self.cache_len), self.dtype) for _ in range(self.n_layers)]
        v_self = [np.zeros((self.heads, 1, self.cache_len, self.dim), self.dtype) for _ in range(self.n_layers)]
        mask = np.full((1, 1, 1, MEAN_DECODE_LEN), MASK_NEG, dtype=self.dtype)
        steps = 0
        for n in range(min(max_tokens, MEAN_DECODE_LEN - 1)):
            mask[0, 0, 0, MEAN_DECODE_LEN - n - 1] = 0.0
            feeds = {"input_ids": np.array([[tokens[n]]], dtype=np.int32),
                     "position_ids": np.array([n], dtype=np.int32),
                     "attention_mask": mask}
            for i in range(self.n_layers):
                feeds[f"k_cache_self_{i}_in"] = k_self[i]
                feeds[f"v_cache_self_{i}_in"] = v_self[i]
            feeds.update(cross)
            outs = self.dec.run(None, feeds)
            names = [o.name for o in self.dec.get_outputs()]
            out = dict(zip(names, outs))
            for i in range(self.n_layers):
                k_self[i] = out[f"k_cache_self_{i}_out"]
                v_self[i] = out[f"v_cache_self_{i}_out"]
            logits = out["logits"].reshape(-1)
            steps += 1
            if n < len(prefix) - 1:
                continue  # still feeding the forced prefix
            nxt = int(np.argmax(logits))
            if nxt == EOT:
                break
            tokens.append(nxt)
        t2 = time.perf_counter()
        text = self.tok.decode([t for t in tokens if t not in self.special], skip_special_tokens=True).strip()
        return Transcript(text, tokens, language or "auto", (t1 - t0) * 1000, (t2 - t1) * 1000, steps, secs)


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(str(path), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    dt = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(raw, dtype=dt).astype(np.float32)
    a = (a - 128) / 128 if sw == 1 else a / float(2 ** (8 * sw - 1))
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr
