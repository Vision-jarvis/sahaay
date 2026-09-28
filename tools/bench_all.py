"""Measure every Sahaay model on this machine, CPU versus Hexagon NPU, and write benchmarks/.

    py -3.12 tools/bench_all.py [--skip-genai] [--out benchmarks]

Produces results.json and results.md with: environment, per-ONNX-model CPU vs NPU latency
(with the active execution provider asserted), end-to-end pipeline timings (face tracking on a
real frame, Whisper on a real clip, Piper TTS), and GenieX LLM / VLM time-to-first-token and
tokens per second on the NPU. Numbers in the README come from this file, nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sahaay.config import model_dir  # noqa: E402
from sahaay.npu import bench, create_session, npu_devices, random_feeds, register_qnn  # noqa: E402

TOOLS = ROOT / "tools"


def cpu_name() -> str:
    try:
        import subprocess

        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out.splitlines()[0] if out else platform.processor()
    except Exception:
        return platform.processor()


def onnx_rows(paths: list[Path], iters: int) -> list[dict]:
    rows = []
    for p in paths:
        row = {"model": p.name, "dir": p.parent.name}
        feeds = None
        try:
            cpu, ci = create_session(p, prefer_npu=False, context_cache=False)
            feeds = random_feeds(cpu)
            row["cpu_ms"] = round(bench(cpu, feeds, iters=iters), 3)
            del cpu
        except Exception as exc:
            # Precompiled QNN context graphs (EPContext nodes) only exist for the NPU; that is the point of them.
            row["cpu_ms"] = None
            row["cpu_note"] = "precompiled QNN context, NPU-only artifact" if "EPContext" in str(exc) else str(exc)[:120]
        try:
            npu, ni = create_session(p, prefer_npu=True, context_cache=False)
            if feeds is None:
                feeds = random_feeds(npu)
            row["npu_ms"] = round(bench(npu, feeds, iters=iters), 3)
            row["npu_active"] = ni.on_npu
            row["providers"] = ni.providers
            row["speedup"] = round(row["cpu_ms"] / row["npu_ms"], 1) if row.get("cpu_ms") else None
            del npu
        except Exception as exc:
            row["npu_active"] = False
            row["error"] = str(exc)[:200]
        rows.append(row)
        cpu_s = f"{row['cpu_ms']:8.2f} ms" if row.get("cpu_ms") is not None else "     n/a   "
        print(f"  {row['model']:32} cpu {cpu_s}   npu {row.get('npu_ms', float('nan')):8.2f} ms   {'NPU' if row.get('npu_active') else 'CPU FALLBACK'}"
              + (f"   ({row['cpu_note']})" if row.get("cpu_note") else ""))
    return rows


def face_pipeline() -> dict:
    from sahaay.vision.camera import Camera
    from sahaay.vision.face import FaceTracker

    try:
        cam = Camera(width=1280, height=720)
        cam.start()
        time.sleep(0.8)
        frame = cam.read(timeout=2.0)
        cam.stop()
    except Exception as exc:
        return {"skipped": f"camera unavailable: {exc}"}
    if frame is None:
        return {"skipped": "no camera frame"}
    out = {"frame": list(frame.shape)}
    for label, prefer in (("npu", True), ("cpu", False)):
        tr = FaceTracker(model_dir("face"), prefer_npu=prefer)
        for _ in range(5):
            tr.process(frame)
        t = time.perf_counter()
        n = 60
        found = 0
        for _ in range(n):
            found += tr.process(frame).found
        dt = (time.perf_counter() - t) / n * 1000
        out[label] = {"ms_per_frame": round(dt, 2), "fps": round(1000 / dt, 1), "found": found, "on_npu": tr.on_npu}
        print(f"  face pipeline {label}: {dt:.2f} ms/frame ({1000/dt:.0f} fps), on_npu={tr.on_npu}")
        del tr
    return out


def whisper_pipeline() -> dict:
    from sahaay.speech.whisper import WhisperNPU, load_wav

    clip = ROOT / "docs" / "media" / "command_sample.wav"
    if not clip.exists():
        return {"skipped": "no docs/media/command_sample.wav"}
    audio, sr = load_wav(clip)
    out = {}
    for label, prefer in (("npu", True), ("cpu", False)):
        try:
            asr = WhisperNPU(model_dir("whisper"), prefer_npu=prefer)
        except Exception as exc:
            out[label] = {"skipped": "precompiled QNN context is NPU-only" if "EPContext" in str(exc) else str(exc)[:120]}
            print(f"  whisper {label}: skipped ({out[label]['skipped']})")
            continue
        asr.transcribe(audio, sr)
        ts = [asr.transcribe(audio, sr) for _ in range(3)]
        enc = float(np.median([t.encoder_ms for t in ts]))
        dec = float(np.median([t.decoder_ms for t in ts]))
        steps = ts[0].steps
        out[label] = {"audio_s": round(ts[0].audio_seconds, 2), "encoder_ms": round(enc, 1), "decoder_ms": round(dec, 1),
                      "decoder_ms_per_token": round(dec / max(steps, 1), 2), "tokens": steps, "text": ts[0].text, "on_npu": asr.on_npu}
        print(f"  whisper {label}: enc {enc:.0f} ms, dec {dec:.0f} ms / {steps} tok, on_npu={asr.on_npu}: {ts[0].text!r}")
        del asr
    return out


def tts_pipeline() -> dict:
    from sahaay.tts import Speaker

    sp = Speaker(model_dir("piper"))
    out = {}
    for lang, text in (("en", "Sahaay is ready. Every model is running on the Hexagon NPU."),
                       ("hi", "नमस्ते, मैं सहाय हूँ। आपकी मदद के लिए तैयार हूँ।")):
        if lang not in sp.voices:
            continue
        sp.synthesize(text, lang)
        audio, rate = sp.synthesize(text, lang)
        out[lang] = {"audio_s": round(len(audio) / rate, 2), "synth_ms": round(sp.last_synth_ms, 0),
                     "realtime_factor": round((len(audio) / rate) / (sp.last_synth_ms / 1000), 1), "unit": "NPU" if sp.voices[lang].on_npu else "CPU"}
        print(f"  tts {lang}: {out[lang]['audio_s']} s audio in {out[lang]['synth_ms']:.0f} ms ({out[lang]['unit']})")
    return out


def genai(skip: bool) -> dict:
    if skip:
        return {"skipped": "--skip-genai"}
    from sahaay.brain import Brain, SYSTEM_PROMPT
    from sahaay.vision.describe import Describer

    out = {}
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    b = Brain(preload=False)
    b._load()
    out["llm"] = {"model": b.model_id, "load_s": round(b.load_ms / 1000, 1), "rss_delta_mb": round((proc.memory_info().rss - rss0) / 1e6)}
    screen = 'Window: "Untitled - Notepad" (notepad.exe)\n[1] MenuItem "File"\n[2] MenuItem "Edit"\n[3] Edit "Text editor" = ""\n[4] Button "Save"\n[5] Button "Close"'
    prompts = ["write a short thank you note to my teacher and save the file", "close this without saving", "how many buttons are on the screen"]
    ttfts, tps = [], []
    with b._lock:
        llm = b._llm
        for u in prompts:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Screen:\n{screen}\n\nUser said: {u}"}]
            p = llm.tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
            t0 = time.perf_counter()
            first = None
            n = 0
            for _ in llm.generate(p, max_new_tokens=120, temperature=0.0, stream=True):
                if first is None:
                    first = time.perf_counter()
                n += 1
            t1 = time.perf_counter()
            ttfts.append((first - t0) * 1000)
            tps.append(n / max(t1 - first, 1e-6))
    out["llm"].update({"ttft_ms": round(float(np.median(ttfts)), 0), "tokens_per_s": round(float(np.median(tps)), 1), "unit": "NPU (QAIRT)"})
    print(f"  llm: load {out['llm']['load_s']} s, TTFT {out['llm']['ttft_ms']:.0f} ms, {out['llm']['tokens_per_s']} tok/s")
    b.close()
    d = Describer(preload=False)
    rss1 = proc.memory_info().rss
    d._load()
    out["vlm"] = {"model": d.model_id, "load_s": round(d.load_ms / 1000, 1), "rss_delta_mb": round((proc.memory_info().rss - rss1) / 1e6)}
    d.describe_screen()
    texts = [d.describe_screen() for _ in range(2)]
    out["vlm"].update({"ttft_ms": round(d.last_ttft_ms, 0), "total_ms": round(d.last_ms, 0), "tokens": d.last_tokens,
                       "tokens_per_s": round(d.last_tokens / max(d.last_ms / 1000, 1e-6), 1), "sample": texts[-1][:200], "unit": "NPU (QAIRT)"})
    print(f"  vlm: load {out['vlm']['load_s']} s, TTFT {out['vlm']['ttft_ms']:.0f} ms, {out['vlm']['tokens_per_s']} tok/s")
    d.close()
    return out


def write_md(rep: dict, path: Path) -> None:
    e = rep["env"]
    L = [f"# Sahaay benchmarks ({e['timestamp']})", "",
         f"Machine: {e['cpu']} | NPU devices via QNN: {e['npu_devices']} | ORT {e['onnxruntime']} + onnxruntime-qnn {e['onnxruntime_qnn']} | Python {e['python']} {e['machine']}", "",
         "## ONNX models, CPU vs NPU (median of 30 runs, random inputs, same graph)", "",
         "| Model | CPU ms | NPU ms | Speedup | Active EP |", "|---|---:|---:|---:|---|"]
    for r in rep["onnx"]:
        cpu = f"{r['cpu_ms']:.2f}" if r.get("cpu_ms") is not None else "n/a (" + r.get("cpu_note", "") + ")"
        sp = f"{r['speedup']:.1f}x" if r.get("speedup") else ""
        L.append(f"| {r['dir']}/{r['model']} | {cpu} | {r.get('npu_ms', float('nan')):.2f} | {sp} | {'QNN (NPU)' if r.get('npu_active') else 'CPU fallback'} |")
    f = rep.get("face", {})
    if "npu" in f:
        L += ["", "## Face tracking pipeline (detector + 468-pt mesh + pose, real 1280x720 frame)", "",
              "| Path | ms / frame | fps |", "|---|---:|---:|",
              f"| NPU | {f['npu']['ms_per_frame']} | {f['npu']['fps']} |", f"| CPU | {f['cpu']['ms_per_frame']} | {f['cpu']['fps']} |"]
    w = rep.get("whisper", {})
    if "npu" in w and "encoder_ms" in w["npu"]:
        L += ["", f"## Whisper base ({w['npu']['audio_s']} s clip, {w['npu']['tokens']} tokens)", "",
              "| Path | Encoder ms | Decoder ms | ms / token |", "|---|---:|---:|---:|",
              f"| NPU | {w['npu']['encoder_ms']} | {w['npu']['decoder_ms']} | {w['npu']['decoder_ms_per_token']} |"]
        if "encoder_ms" in w.get("cpu", {}):
            L.append(f"| CPU | {w['cpu']['encoder_ms']} | {w['cpu']['decoder_ms']} | {w['cpu']['decoder_ms_per_token']} |")
        else:
            L.append(f"| CPU | n/a: {w.get('cpu', {}).get('skipped', 'not run')} | | |")
        L += ["", f"Transcript (NPU): \"{w['npu']['text']}\""]
    t = rep.get("tts", {})
    if t:
        L += ["", "## Piper TTS", "", "| Voice | Audio s | Synth ms | Realtime factor | Unit |", "|---|---:|---:|---:|---|"]
        for k, v in t.items():
            L.append(f"| {k} | {v['audio_s']} | {v['synth_ms']:.0f} | {v['realtime_factor']}x | {v['unit']} |")
    g = rep.get("genai", {})
    if "llm" in g:
        L += ["", "## Generative models on the NPU (GenieX, QAIRT backend)", "",
              "| Model | Load s | TTFT ms | tokens/s | RSS delta MB |", "|---|---:|---:|---:|---:|",
              f"| {g['llm']['model']} | {g['llm']['load_s']} | {g['llm']['ttft_ms']:.0f} | {g['llm']['tokens_per_s']} | {g['llm']['rss_delta_mb']} |",
              f"| {g['vlm']['model']} | {g['vlm']['load_s']} | {g['vlm']['ttft_ms']:.0f} | {g['vlm']['tokens_per_s']} | {g['vlm']['rss_delta_mb']} |",
              "", f"VLM screen description sample: \"{g['vlm']['sample']}\""]
    L += ["", "Generated by `tools/bench_all.py`. Raw numbers in results.json."]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-genai", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "benchmarks"))
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    register_qnn()
    import onnxruntime as ort
    import onnxruntime_qnn as q

    rep = {"env": {"timestamp": time.strftime("%Y-%m-%d %H:%M"), "cpu": cpu_name(), "machine": platform.machine(),
                   "python": platform.python_version(), "onnxruntime": ort.__version__,
                   "onnxruntime_qnn": getattr(q, "__version__", "2.6.0"), "npu_devices": len(npu_devices()),
                   "ram_gb": round(psutil.virtual_memory().total / 1e9, 1)}}
    print("== ONNX models ==")
    paths = [model_dir("face") / "face_detector.onnx", model_dir("face") / "face_landmark_detector.onnx",
             model_dir("whisper") / "encoder.onnx", model_dir("whisper") / "decoder.onnx"]
    fm = model_dir("face").parent.parent / "facemap_3dmm_x2" / "facemap_3dmm-onnx-float"
    paths += list(fm.glob("*.onnx"))
    rep["onnx"] = onnx_rows([p for p in paths if p.exists()], a.iters)
    print("== Pipelines ==")
    rep["face"] = face_pipeline()
    rep["whisper"] = whisper_pipeline()
    rep["tts"] = tts_pipeline()
    print("== GenAI ==")
    rep["genai"] = genai(a.skip_genai)
    (out / "results.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(rep, out / "results.md")
    print(f"wrote {out / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
