"""Transcribe WAV files (or record from the mic) with Whisper on the NPU.

    py -3.12 tools/asr_demo.py file1.wav [file2.wav ...] [--lang en|hi] [--cpu]
    py -3.12 tools/asr_demo.py --mic 5          # record 5 s then transcribe
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sahaay.speech.whisper import SAMPLE_RATE, WhisperNPU, load_wav  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "aihub" / "whisper_base_x2" / "whisper_base-precompiled_qnn_onnx-float-qualcomm_snapdragon_x2_elite"


def main() -> int:
    args = sys.argv[1:]
    lang = "en"
    if "--lang" in args:
        i = args.index("--lang")
        lang = args[i + 1]
        del args[i : i + 2]
    prefer_npu = "--cpu" not in args
    args = [a for a in args if a != "--cpu"]
    t0 = time.perf_counter()
    asr = WhisperNPU(MODEL_DIR, prefer_npu=prefer_npu)
    print(f"loaded in {time.perf_counter()-t0:.1f}s | encoder providers {asr.enc_info.providers} | decoder providers {asr.dec_info.providers}")
    clips: list[tuple[str, np.ndarray, int]] = []
    if args and args[0] == "--mic":
        secs = float(args[1]) if len(args) > 1 else 5.0
        import sounddevice as sd

        print(f"recording {secs}s ... speak now")
        audio = sd.rec(int(secs * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        clips.append(("mic", audio[:, 0], SAMPLE_RATE))
    else:
        for p in args:
            a, sr = load_wav(p)
            clips.append((Path(p).name, a, sr))
    for name, a, sr in clips:
        r = asr.transcribe(a, sr, language=lang)
        print(f"[{name}] {r.audio_seconds:.1f}s audio | encoder {r.encoder_ms:.0f} ms | decoder {r.decoder_ms:.0f} ms for {r.steps} steps ({r.decoder_ms/max(r.steps,1):.1f} ms/step)")
        print(f"   -> {r.text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
