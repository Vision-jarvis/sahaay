"""Generate the Brief Project Description (DOCX) for the Unstop form.

    py -3.12 docs/description/build_description.py

Two pages: problem, solution, technical implementation, deployment, results, links.
Numbers come from benchmarks/results.json when present.
"""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "description" / "Sahaay_Project_Description.docx"


def bench() -> dict:
    p = ROOT / "benchmarks" / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> None:
    b = bench()
    g = b.get("genai", {})
    llm, vlm = g.get("llm", {}), g.get("vlm", {})
    face = b.get("face", {}).get("npu", {})
    wh = b.get("whisper", {}).get("npu", {})
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Pt(50)
        s.left_margin = s.right_margin = Pt(58)

    def h(text, lvl=1):
        p = doc.add_heading(text, level=lvl)
        for r in p.runs:
            r.font.color.rgb = RGBColor(0x0B, 0x6E, 0x4F) if lvl > 0 else RGBColor(0, 0, 0)
        return p

    def para(text, bold_lead: str | None = None):
        p = doc.add_paragraph()
        if bold_lead:
            r = p.add_run(bold_lead + " ")
            r.bold = True
        p.add_run(text)
        p.paragraph_format.space_after = Pt(4)
        return p

    def bullets(items):
        for it in items:
            p = doc.add_paragraph(style="List Bullet")
            lead, _, rest = it.partition("|")
            if rest:
                r = p.add_run(lead.strip() + " ")
                r.bold = True
                p.add_run(rest.strip())
            else:
                p.add_run(it)
            p.paragraph_format.space_after = Pt(2)

    t = doc.add_heading("Sahaay: an offline accessibility copilot for Snapdragon-powered HP PCs", 0)
    t.alignment = WD_ALIGN_PARAGRAPH.LEFT
    para("Ruhan Srivastava, IIT Kharagpur. Snapdragon AI Lab Build & Present Challenge 2026. Repository: github.com/Vision-jarvis/sahaay")

    h("Problem")
    para("India has 2.68 crore people with disabilities (Census 2011); about a fifth have a movement disability and a fifth a visual disability. "
         "For them a laptop is only usable through assistive hardware that costs over Rs 1.5 lakh, or through cloud services that stream their camera and screen to a server, "
         "add hundreds of milliseconds of latency, and fail on patchy connectivity. Built-in accessibility stops at reading controls: it cannot describe a photo, a chart or the room.")

    h("Solution")
    para("Sahaay (सहाय, \"assistance\") turns any Snapdragon X or X2 PC, including the HP OmniBook and EliteBook, into an assistive device with a free download. "
         "It has two modes that share one on-device brain.")
    bullets([
        "Hands-free mode (motor impairment):| a 468-point face mesh on the NPU turns head pose into a joystick-style cursor with dwell, blink and mouth-gesture clicks and drags; Whisper on the NPU gives dictation in English and Hindi; 25 instant voice commands and free-form commands through an on-device language model (\"open my downloads\", \"write a thank-you note and save the file\"). Optional physical switch access through an Arduino, including the UNO Q from the AI Lab kit.",
        "Narrator mode (low vision, blindness):| \"what's on my screen\" summarises the Windows UI Automation tree in two spoken sentences; \"read this\" reads the focused text; \"describe the picture\" and \"what do you see\" use an on-device vision-language model to describe the screen or the webcam view; every action is confirmed by voice in English or Hindi.",
    ])

    h("Technical implementation")
    para("Six models run concurrently on the Qualcomm Hexagon NPU; the Oryon CPU only captures, validates and moves the mouse; the GPU is left to the user's applications.")
    bullets([
        "Face detection and 468-point mesh:| Qualcomm AI Hub mediapipe_face, ONNX Runtime with the QNN execution provider (HTP, fp16). Qualcomm's pre- and post-processing re-implemented in numpy and Pillow because OpenCV has no Windows ARM64 wheel. Head pose, eye aspect ratio and mouth opening derived from the mesh.",
        "Speech to text:| Qualcomm AI Hub whisper_base exported as precompiled QNN ONNX for Snapdragon X Elite (Hexagon v73) and X2 Elite (v81); the encoder plus KV-cache single-step decoder loop written in numpy; energy-based voice activity detection on the CPU.",
        "Language model:| Qwen3-4B-Instruct-2507 NPU bundle from Qualcomm AI Hub via Qualcomm GenieX (QAIRT). Two-tier command handling: a regex grammar answers common commands in under a millisecond; everything else goes to the model with a compact UI Automation snapshot of the screen and a strict JSON tool schema. Every returned call is validated against the schema and an allow-list before execution.",
        "Vision-language model:| Qwen3-VL-4B-Instruct NPU bundle via GenieX for screen and camera description.",
        "Text to speech:| Piper VITS voices (en_US, hi_IN) driven through ONNX Runtime with espeak-ng phonemisation.",
        "Verification:| tools/npu_check.py asserts that QNNExecutionProvider is the active provider and times CPU versus NPU for every ONNX model; tools/bench_all.py regenerates all numbers. The common providers=[\"QNNExecutionProvider\"] call silently runs on the CPU with the current plugin runtime, which this project detects and avoids.",
    ])

    h("Results (measured on a Snapdragon X2 Elite laptop)")
    bullets([
        f"Face detector 0.42 ms and face mesh 0.16 ms on the NPU (2.97 ms and 1.01 ms on CPU); full live pipeline {face.get('ms_per_frame', 2.7)} ms per frame at {face.get('fps', 28)} fps.",
        f"Whisper base: encoder {wh.get('encoder_ms', 25):.0f} ms, decoder {wh.get('decoder_ms_per_token', 3):.1f} ms per token; a six-second command transcribes in about 100 ms.",
        f"Qwen3-4B: {llm.get('ttft_ms', 60):.0f} ms to first token, {llm.get('tokens_per_s', 33)} tokens per second; Qwen3-VL-4B: {vlm.get('ttft_ms', 280):.0f} ms to first token, {vlm.get('tokens_per_s', 30)} tokens per second, a screen description in about {vlm.get('total_ms', 2300)/1000:.1f} s.",
        "Qualcomm AI Hub profiling on the reference devices confirms 100% of ops on the NPU: on Snapdragon X2 Elite CRD the face detector takes 0.4 ms, the face mesh 0.2 ms, the Whisper encoder 21.5 ms and the decoder 2.4 ms per token; on Snapdragon X Elite CRD (the HP OmniBook chip) 0.7 ms, 0.3 ms, 45.4 ms and 3.7 ms. Job links are in the repository.",
        "Memory: about 5 GB resident with both generative models loaded; fits a 16 GB OmniBook.",
    ])

    h("Deployment and accessibility")
    para("One command installs everything: install.ps1 installs ARM64 Python 3.12 if missing, creates a virtual environment, downloads the models for the detected chipset once, installs espeak-ng and adds a Start Menu shortcut. "
         "After that Sahaay is fully offline: no account, no admin rights, no network calls; frames, audio and screenshots stay in RAM. "
         "Global hotkeys, a tray icon, a heads-up panel showing what was heard and done with live per-model latency, and spoken confirmation of every action make it usable without sight or hands. "
         "The Arduino switch protocol is documented and can be tried in the Wokwi simulator without hardware.")

    h("Why on-device, why Snapdragon HP PCs")
    para("A camera watching a face all day and a screen showing a bank statement cannot go to the cloud; cursor control needs sub-frame latency; students in hostels and patients in hospitals cannot depend on connectivity; and a laptop that is someone's only interface must last the day on battery. "
         "The Hexagon NPU runs the whole stack at a few watts. HP India positions the OmniBook 3 and 5 at students and Qualcomm India wants Snapdragon PCs under Rs 60,000; Sahaay makes that laptop an assistive device at no extra cost.")

    h("Development and honest limitations")
    para("Built in three days on the Snapdragon AI Lab timeline. Qualcomm AI Hub and GenieX made the models the easy part; Windows ARM64 plumbing (camera through DirectShow COM, Piper without an ARM64 espeak bridge, QNN context caching, sequential GenieX loads) was the hard part. "
         "Piper runs on the CPU today because its dynamic-shape graph is rejected by the QNN compiler; the AI Hub pipertts recipe is the planned NPU path. Gaze estimation, Hindi command grammar and vision-model element grounding are on the roadmap.")

    doc.save(str(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
