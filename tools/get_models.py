"""First-run model fetcher. Everything Sahaay needs, downloaded once, then fully offline.

    py -3.12 tools/get_models.py [--dest DIR] [--chipset x2-elite|x-elite|auto] [--no-genai]

Sources
- Face detector + landmarks, Whisper base: Qualcomm AI Hub exports for the detected Snapdragon
  chipset, packaged as a zip on the Sahaay GitHub Release (SAHAAY_MODELS_URL overrides).
- Whisper tokenizer + feature extractor: Hugging Face openai/whisper-base (config files only).
- Piper voices (en_US lessac, hi_IN pratham): rhasspy/piper-voices on Hugging Face.
- Qwen3-4B-Instruct-2507 and Qwen3-VL-4B-Instruct NPU bundles: Qualcomm AI Hub via GenieX.
- espeak-ng phonemizer: winget (eSpeak-NG.eSpeak-NG).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sahaay.config import USER_DIR  # noqa: E402

RELEASE = os.environ.get("SAHAAY_MODELS_URL", "https://github.com/Vision-jarvis/sahaay/releases/download/models-v1")
PIPER = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
VOICES = {"en_US-lessac-medium": "en/en_US/lessac/medium", "hi_IN-pratham-medium": "hi/hi_IN/pratham/medium"}
GENIEX_BUNDLES = ["ai-hub-models/Qwen3-4B-Instruct-2507", "ai-hub-models/Qwen3-VL-4B-Instruct"]


def detect_chipset() -> str:
    try:
        name = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                              capture_output=True, text=True, timeout=20).stdout.lower()
    except Exception:
        name = ""
    if "x2" in name:
        return "x2-elite"
    return "x-elite"  # X Elite, X Plus and X share Hexagon v73 binaries


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return
    print(f"  fetching {url}")
    req = Request(url, headers={"User-Agent": "sahaay-get-models"})
    with urlopen(req) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(USER_DIR / "models"))
    ap.add_argument("--chipset", default="auto", choices=["auto", "x2-elite", "x-elite"])
    ap.add_argument("--no-genai", action="store_true", help="skip the 5 GB LLM/VLM bundles")
    a = ap.parse_args()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    chip = detect_chipset() if a.chipset == "auto" else a.chipset
    print(f"chipset: {chip}   models dir: {dest}")

    print("[1/5] AI Hub vision + speech models")
    z = dest / f"aihub-{chip}.zip"
    download(f"{RELEASE}/aihub-{chip}.zip", z)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dest)
    z.unlink(missing_ok=True)

    print("[2/5] Whisper tokenizer and feature extractor")
    hf = dest / "whisper_base" / "hf"
    if not (hf / "tokenizer.json").exists():
        from transformers import WhisperFeatureExtractor, WhisperTokenizer

        WhisperTokenizer.from_pretrained("openai/whisper-base").save_pretrained(hf)
        WhisperFeatureExtractor.from_pretrained("openai/whisper-base").save_pretrained(hf)
    print("  ok")

    print("[3/5] Piper voices")
    for name, sub in VOICES.items():
        for ext in ("onnx", "onnx.json"):
            download(f"{PIPER}/{sub}/{name}.{ext}", dest / "piper" / f"{name}.{ext}")

    print("[4/5] espeak-ng phonemizer")
    if not any(Path(p).exists() for p in (r"C:\Program Files\eSpeak NG\espeak-ng.exe", r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe")) and not shutil.which("espeak-ng"):
        subprocess.run(["winget", "install", "--id", "eSpeak-NG.eSpeak-NG", "--exact", "--silent",
                        "--accept-source-agreements", "--accept-package-agreements"], check=False)
    print("  ok")

    print("[5/5] Language and vision models on the NPU (GenieX bundles from Qualcomm AI Hub)")
    if a.no_genai:
        print("  skipped")
    else:
        geniex = Path(sys.executable).parent / "Scripts" / "geniex-py.exe"
        cmd = [str(geniex)] if geniex.exists() else [sys.executable, "-m", "geniex"]
        for b in GENIEX_BUNDLES:
            print(f"  pulling {b}")
            subprocess.run(cmd + ["pull", b], check=False)
    print("done. Run:  run.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
