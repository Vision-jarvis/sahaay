"""Generate the Sahaay pitch deck (PPTX) from one content model.

    py -3.12 docs/deck/build_deck.py [--out docs/deck/Sahaay_Pitch.pptx]

PDF export is done separately with PowerPoint (docs/deck/export_pdf.ps1) because the challenge
form wants both formats. Numbers come from benchmarks/results.json when present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "docs" / "media"
BG = RGBColor(0x10, 0x14, 0x18)
PANEL = RGBColor(0x18, 0x1E, 0x25)
FG = RGBColor(0xE8, 0xED, 0xF2)
DIM = RGBColor(0x8A, 0x97, 0xA6)
ACCENT = RGBColor(0x3D, 0xDC, 0x97)
WARN = RGBColor(0xFF, 0xB4, 0x54)
BLUE = RGBColor(0x7E, 0xC8, 0xFF)
W, H = Inches(13.333), Inches(7.5)


def load_bench() -> dict:
    p = ROOT / "benchmarks" / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    # ---- primitives ----------------------------------------------------------------------
    def slide(self, title: str | None = None, kicker: str | None = None):
        s = self.prs.slides.add_slide(self.blank)
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = BG
        self.n += 1
        if kicker:
            self.text(s, kicker, Inches(0.6), Inches(0.35), Inches(8), Inches(0.4), size=12, color=ACCENT, bold=True)
        if title:
            self.text(s, title, Inches(0.6), Inches(0.6), Inches(12), Inches(0.9), size=30, color=FG, bold=True)
        self.text(s, f"Sahaay  ·  {self.n}", Inches(11.3), Inches(7.0), Inches(1.8), Inches(0.3), size=10, color=DIM, align=PP_ALIGN.RIGHT)
        return s

    def text(self, s, txt, x, y, w, h, *, size=16, color=FG, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Segoe UI"):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(0.05)
        lines = txt if isinstance(txt, list) else [txt]
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            r = p.add_run()
            r.text = line
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = color
            r.font.name = font
        return tb

    def bullets(self, s, items, x, y, w, h, *, size=16, color=FG, gap=6):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        for i, it in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(gap)
            bold_part, _, rest = it.partition("|")
            r = p.add_run()
            r.text = "•  " + bold_part.strip()
            r.font.size = Pt(size)
            r.font.bold = bool(rest)
            r.font.color.rgb = ACCENT if rest else color
            r.font.name = "Segoe UI"
            if rest:
                r2 = p.add_run()
                r2.text = "  " + rest.strip()
                r2.font.size = Pt(size)
                r2.font.color.rgb = color
                r2.font.name = "Segoe UI"
        return tb

    def box(self, s, txt, x, y, w, h, *, fill=PANEL, color=FG, size=13, line=None, bold=False, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
        sh = s.shapes.add_shape(shape, x, y, w, h)
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
        if line:
            sh.line.color.rgb = line
            sh.line.width = Pt(1.5)
        else:
            sh.line.fill.background()
        tf = sh.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = Inches(0.08)
        lines = txt if isinstance(txt, list) else [txt]
        for i, line_txt in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = line_txt
            r.font.size = Pt(size if i == 0 else max(size - 3, 9))
            r.font.bold = bold if i == 0 else False
            r.font.color.rgb = color if i == 0 else DIM
            r.font.name = "Segoe UI"
        return sh

    def arrow(self, s, x1, y1, x2, y2, color=DIM):
        c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
        c.line.color.rgb = color
        c.line.width = Pt(1.75)
        ln = c.line._get_or_add_ln()
        from pptx.oxml.ns import qn

        tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
        ln.append(tail)
        return c

    def table(self, s, rows, x, y, w, col_widths=None, *, size=12, header=True, row_h=Inches(0.38)):
        nrows, ncols = len(rows), len(rows[0])
        gt = s.shapes.add_table(nrows, ncols, x, y, w, row_h * nrows).table
        if col_widths:
            for i, cw in enumerate(col_widths):
                gt.columns[i].width = cw
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                cell = gt.cell(r, c)
                cell.fill.solid()
                cell.fill.fore_color.rgb = PANEL if not (header and r == 0) else RGBColor(0x22, 0x2A, 0x33)
                cell.margin_left = cell.margin_right = Inches(0.06)
                cell.margin_top = cell.margin_bottom = Inches(0.03)
                tf = cell.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                run = p.add_run()
                run.text = str(val)
                run.font.size = Pt(size)
                run.font.name = "Segoe UI"
                run.font.bold = header and r == 0
                run.font.color.rgb = ACCENT if (header and r == 0) else FG
        return gt

    def image(self, s, path: Path, x, y, w=None, h=None, placeholder: str | None = None):
        if path.exists():
            return s.shapes.add_picture(str(path), x, y, width=w, height=h)
        return self.box(s, placeholder or path.name, x, y, w or Inches(4), h or Inches(2.5), fill=PANEL, color=DIM, size=12, line=DIM)

    def save(self, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(out))


def build(out: Path) -> None:
    b = load_bench()
    g = b.get("genai", {})
    llm = g.get("llm", {})
    vlm = g.get("vlm", {})
    face = b.get("face", {}).get("npu", {})
    wh = b.get("whisper", {}).get("npu", {})
    onnx = {r["model"]: r for r in b.get("onnx", [])}

    def ms(name, key, default):
        return f"{onnx[name][key]:.2f}" if name in onnx and key in onnx[name] else default

    d = Deck()

    # 1 Title
    s = d.slide()
    d.text(s, "Sahaay", Inches(0.8), Inches(2.0), Inches(8), Inches(1.4), size=66, color=ACCENT, bold=True)
    d.text(s, "सहाय  ·  \"assistance\"", Inches(0.85), Inches(3.25), Inches(8), Inches(0.5), size=18, color=DIM)
    d.text(s, "Your PC, hands-free and eyes-free.", Inches(0.8), Inches(3.9), Inches(11), Inches(0.8), size=32, color=FG, bold=True)
    d.text(s, "An offline accessibility copilot for Snapdragon-powered HP PCs. Six AI models, all on the Hexagon NPU, nothing leaves the laptop.",
           Inches(0.8), Inches(4.7), Inches(11.5), Inches(0.9), size=18, color=FG)
    d.text(s, ["Ruhan Srivastava  ·  Snapdragon AI Lab Build & Present Challenge 2026", "github.com/Vision-jarvis/sahaay"],
           Inches(0.8), Inches(6.2), Inches(11), Inches(0.8), size=14, color=DIM)

    # 2 Problem
    s = d.slide("A laptop is useless if you cannot move a mouse or see a screen", kicker="THE PROBLEM")
    d.bullets(s, [
        "2.68 crore Indians live with a disability (Census 2011).| Roughly one in five with a movement disability, one in five with a visual disability.",
        "Assistive hardware is priced like medical equipment.| Eye-gaze systems cost over Rs 1.5 lakh, need a dongle, a licence and often the cloud.",
        "Cloud AI cannot do this job.| A camera watching your face all day and a screen showing your bank statement cannot be streamed to a server. Latency and patchy 4G make it unusable anyway.",
        "Built-in accessibility stops at the basics.| Windows Narrator reads controls; it cannot describe a photo, a chart, or the room. Voice Access needs the cloud for anything free-form.",
    ], Inches(0.7), Inches(1.7), Inches(7.6), Inches(5), size=16, gap=12)
    d.box(s, ["Rs 60,000", "HP OmniBook 3 with a 45 TOPS NPU"], Inches(8.9), Inches(1.9), Inches(3.8), Inches(1.4), fill=PANEL, color=ACCENT, size=30, bold=True)
    d.box(s, ["Rs 0", "Sahaay, forever offline"], Inches(8.9), Inches(3.5), Inches(3.8), Inches(1.4), fill=PANEL, color=ACCENT, size=30, bold=True)
    d.box(s, ["Rs 1,50,000+", "a commercial eye-gaze system"], Inches(8.9), Inches(5.1), Inches(3.8), Inches(1.4), fill=PANEL, color=WARN, size=26, bold=True)

    # 3 What it does
    s = d.slide("One app, two people it changes everything for", kicker="WHAT SAHAAY DOES")
    d.box(s, "Hands-free mode", Inches(0.7), Inches(1.6), Inches(5.9), Inches(0.5), fill=ACCENT, color=BG, size=16, bold=True)
    d.bullets(s, [
        "Head cursor.| Nudge your head, the cursor moves; hold still, it stays. Auto-calibrates.",
        "Clicks without hands.| Dwell ring, short blink = left click, long blink = right, open mouth = drag.",
        "Dictation anywhere,| English and Hindi.",
        "25 instant commands| in under a millisecond: click, scroll, press ctrl s, open Chrome, next window.",
        "Free-form commands| through the on-device language model: \"open my downloads\", \"write a thank-you note and save it\".",
        "Physical switches| via an Arduino for people who cannot use the camera.",
    ], Inches(0.7), Inches(2.2), Inches(5.9), Inches(4.8), size=14, gap=8)
    d.box(s, "Narrator mode", Inches(6.9), Inches(1.6), Inches(5.9), Inches(0.5), fill=BLUE, color=BG, size=16, bold=True)
    d.bullets(s, [
        "\"What's on my screen?\"| Two spoken sentences: which app, what content, which buttons matter.",
        "\"Read this.\"| Reads the focused paragraph or field.",
        "\"Describe the picture.\"| The vision-language model describes what accessibility trees cannot: photos, charts, canvases.",
        "\"What do you see?\"| Describes the webcam view, reads labels and signs.",
        "Every action is spoken back,| so a blind user always knows what just happened.",
        "Voices in English and Hindi,| generated on the laptop.",
    ], Inches(6.9), Inches(2.2), Inches(5.9), Inches(4.8), size=14, gap=8)

    # 4 Demo
    s = d.slide("Demo", kicker="SEE IT")
    d.image(s, MEDIA / "demo_handsfree_still.png", Inches(0.7), Inches(1.6), w=Inches(6), placeholder="Hands-free demo (GIF in the README)")
    d.image(s, MEDIA / "demo_narrator_still.png", Inches(6.9), Inches(1.6), w=Inches(6), placeholder="Narrator demo (GIF in the README)")
    d.text(s, "Head moves the cursor, a blink clicks, \"open notepad and type a thank-you note to my teacher\" does the rest.",
           Inches(0.7), Inches(5.3), Inches(6), Inches(0.9), size=13, color=DIM)
    d.text(s, "\"What's on my screen?\"  \"Describe the picture.\"  \"What do you see?\"  Answered in about two seconds, on the NPU.",
           Inches(6.9), Inches(5.3), Inches(6), Inches(0.9), size=13, color=DIM)
    d.text(s, "Full videos and GIFs: github.com/Vision-jarvis/sahaay", Inches(0.7), Inches(6.3), Inches(12), Inches(0.4), size=13, color=ACCENT)

    # 5 Architecture
    s = d.slide("Six models, one NPU, the CPU only moves the mouse", kicker="ARCHITECTURE")
    y0 = Inches(1.7)
    col = Inches(2.35)
    d.box(s, ["Webcam", "1280x720, 30 fps"], Inches(0.5), y0, col, Inches(0.8), fill=PANEL)
    d.box(s, ["Microphone", "16 kHz, VAD on CPU"], Inches(0.5), y0 + Inches(1.1), col, Inches(0.8), fill=PANEL)
    d.box(s, ["Screen", "UI Automation tree + screenshot"], Inches(0.5), y0 + Inches(2.2), col, Inches(0.8), fill=PANEL)
    d.box(s, ["Arduino switches", "JSON over USB serial (optional)"], Inches(0.5), y0 + Inches(3.3), col, Inches(0.8), fill=PANEL)
    nx = Inches(3.4)
    d.box(s, ["Face detector + 468-pt mesh", "0.4 + 0.2 ms · head pose, blink, mouth"], nx, y0, Inches(3.1), Inches(0.8), fill=PANEL, line=ACCENT)
    d.box(s, ["Whisper base", "encoder 25 ms, decoder 3 ms/token"], nx, y0 + Inches(1.1), Inches(3.1), Inches(0.8), fill=PANEL, line=ACCENT)
    d.box(s, ["Qwen3-4B-Instruct", "screen list + speech to JSON tool calls, 33 tok/s"], nx, y0 + Inches(2.2), Inches(3.1), Inches(0.8), fill=PANEL, line=ACCENT)
    d.box(s, ["Qwen3-VL-4B-Instruct", "describes screen and camera, 30 tok/s"], nx, y0 + Inches(3.3), Inches(3.1), Inches(0.8), fill=PANEL, line=ACCENT)
    d.text(s, "HEXAGON NPU", nx, y0 - Inches(0.4), Inches(3.1), Inches(0.35), size=12, color=ACCENT, bold=True, align=PP_ALIGN.CENTER)
    cx = Inches(7.1)
    d.box(s, ["Head cursor + gestures", "joystick, dwell, blink, drag · Win32"], cx, y0, Inches(2.9), Inches(0.8), fill=PANEL, line=DIM)
    d.box(s, ["Fast grammar", "25 commands, <1 ms"], cx, y0 + Inches(1.1), Inches(2.9), Inches(0.8), fill=PANEL, line=DIM)
    d.box(s, ["Action executor", "click, type, keys, open, focus, scroll, read"], cx, y0 + Inches(2.2), Inches(2.9), Inches(0.8), fill=PANEL, line=DIM)
    d.box(s, ["Schema validator + allow-list", "no invented tools or elements"], cx, y0 + Inches(3.3), Inches(2.9), Inches(0.8), fill=PANEL, line=DIM)
    d.text(s, "ORYON CPU", cx, y0 - Inches(0.4), Inches(2.9), Inches(0.35), size=12, color=DIM, bold=True, align=PP_ALIGN.CENTER)
    ox = Inches(10.6)
    d.box(s, ["Windows", "any app, any control"], ox, y0 + Inches(0.55), Inches(2.3), Inches(0.9), fill=PANEL)
    d.box(s, ["Piper TTS en / hi", "16x real time"], ox, y0 + Inches(2.2), Inches(2.3), Inches(0.8), fill=PANEL)
    d.box(s, ["HUD + tray", "live per-model latency and compute unit"], ox, y0 + Inches(3.3), Inches(2.3), Inches(0.8), fill=PANEL)
    for i in range(4):
        d.arrow(s, Inches(0.5) + col, y0 + Inches(1.1) * i + Inches(0.4), nx, y0 + Inches(1.1) * i + Inches(0.4))
        d.arrow(s, nx + Inches(3.1), y0 + Inches(1.1) * i + Inches(0.4), cx, y0 + Inches(1.1) * i + Inches(0.4))
    d.arrow(s, cx + Inches(2.9), y0 + Inches(0.4), ox, y0 + Inches(1.0))
    d.arrow(s, cx + Inches(2.9), y0 + Inches(2.6), ox, y0 + Inches(1.0))
    d.arrow(s, cx + Inches(2.9), y0 + Inches(2.6), ox, y0 + Inches(2.6))
    d.text(s, "Runtimes: ONNX Runtime + Qualcomm QNN execution provider (HTP) for face, speech and TTS graphs · Qualcomm GenieX (QAIRT) for the language and vision-language models · all models from Qualcomm AI Hub",
           Inches(0.5), Inches(6.35), Inches(12.3), Inches(0.7), size=12, color=DIM)

    # 6 Models
    s = d.slide("Models implemented", kicker="TECHNICAL IMPLEMENTATION")
    rows = [["Job", "Model (Qualcomm AI Hub)", "Precision", "Runtime", "Unit"],
            ["Face detection", "mediapipe_face detector (BlazeFace)", "fp16 on HTP", "ONNX Runtime QNN EP", "NPU"],
            ["468-pt face mesh, pose, blink", "mediapipe_face landmark detector", "fp16 on HTP", "ONNX Runtime QNN EP", "NPU"],
            ["Speech to text, en + hi", "whisper_base encoder + KV-cache decoder", "fp16, precompiled QNN", "ONNX Runtime QNN EP", "NPU"],
            ["Intent to tool calls, summaries", "qwen3_4b_instruct_2507 NPU bundle", "4-bit weights", "GenieX (QAIRT)", "NPU"],
            ["Screen and camera description", "qwen3_vl_4b_instruct NPU bundle", "4-bit weights", "GenieX (QAIRT)", "NPU"],
            ["Text to speech, en + hi", "Piper VITS (rhasspy voices)", "fp32", "ONNX Runtime", "CPU"]]
    d.table(s, rows, Inches(0.6), Inches(1.7), Inches(12.1), [Inches(2.7), Inches(3.9), Inches(1.9), Inches(2.2), Inches(1.4)], size=12, row_h=Inches(0.45))
    d.bullets(s, [
        "Exported with qai-hub-models for both Snapdragon X2 Elite (Hexagon v81, dev laptop) and Snapdragon X Elite (Hexagon v73, the OmniBook).| The installer detects the chipset.",
        "No fine-tuning needed.| Qualcomm's pre- and post-processing for MediaPipe and Whisper re-implemented in numpy so the app runs on ARM64 Windows without torch or OpenCV.",
        "Proof, not claims.| tools/npu_check.py asserts QNNExecutionProvider is active before any NPU number is reported. The common providers=[\"QNNExecutionProvider\"] pattern silently runs on the CPU with today's plugin runtime.",
    ], Inches(0.6), Inches(5.1), Inches(12.1), Inches(2), size=13, gap=6)

    # 7 Compute cores + performance
    s = d.slide("Measured on the NPU, not estimated", kicker="PERFORMANCE")
    rows = [["Stage", "CPU", "NPU", "Note"],
            ["Face detector 256x256", ms("face_detector.onnx", "cpu_ms", "2.97") + " ms", ms("face_detector.onnx", "npu_ms", "0.42") + " ms", "7x"],
            ["Face mesh 192x192", ms("face_landmark_detector.onnx", "cpu_ms", "1.01") + " ms", ms("face_landmark_detector.onnx", "npu_ms", "0.16") + " ms", "6x"],
            ["Full face pipeline, live 720p", "", f"{face.get('ms_per_frame', 2.7)} ms/frame", f"{face.get('fps', 28)} fps, camera-bound"],
            ["Whisper encoder (30 s window)", "", f"{wh.get('encoder_ms', 25):.0f} ms", ""],
            ["Whisper decoder", "", f"{wh.get('decoder_ms_per_token', 3):.1f} ms/token", "a 6 s command in ~100 ms"],
            ["Qwen3-4B (GenieX)", "", f"{llm.get('ttft_ms', 60):.0f} ms to first token, {llm.get('tokens_per_s', 33)} tok/s", f"loads in {llm.get('load_s', 8)} s"],
            ["Qwen3-VL-4B screen description", "", f"{vlm.get('ttft_ms', 280):.0f} ms to first token, {vlm.get('tokens_per_s', 30)} tok/s", f"{vlm.get('total_ms', 2300)/1000:.1f} s per description"],
            ["Piper TTS, 4 s of speech", "230 ms", "", "16x real time"]]
    d.table(s, rows, Inches(0.6), Inches(1.6), Inches(7.6), [Inches(2.8), Inches(1.1), Inches(2.3), Inches(1.4)], size=11, row_h=Inches(0.42))
    d.text(s, "Compute cores used", Inches(8.6), Inches(1.6), Inches(4.2), Inches(0.4), size=16, color=ACCENT, bold=True)
    d.bullets(s, [
        "NPU:| face detector, face mesh, Whisper encoder and decoder, Qwen3-4B, Qwen3-VL-4B. Everything that is a neural network.",
        "CPU:| capture, VAD, mel features, UI Automation, grammar, validation, cursor maths, SendInput, overlay, TTS.",
        "GPU:| deliberately nothing; the user's apps keep it.",
        "Concurrent:| the cursor stays at 28 fps and Whisper keeps listening while the LLM plans and the VLM describes, all on one NPU.",
    ], Inches(8.6), Inches(2.1), Inches(4.3), Inches(4), size=12, gap=8)
    d.text(s, ["Qualcomm AI Hub profiling, every op on the NPU (job links in benchmarks/aihub_profiles.md):",
               "Snapdragon X2 Elite CRD: face detector 0.4 ms, face mesh 0.2 ms, Whisper encoder 21.5 ms, decoder 2.4 ms/token.",
               "Snapdragon X Elite CRD (HP OmniBook): face detector 0.7 ms, face mesh 0.3 ms, Whisper encoder 45.4 ms, decoder 3.7 ms/token."],
           Inches(0.6), Inches(5.55), Inches(7.6), Inches(1.4), size=11, color=DIM)

    # 8 Deployment
    s = d.slide("Installs in one command, then never needs the internet again", kicker="DEPLOYMENT & ACCESSIBILITY")
    d.box(s, ["powershell -ExecutionPolicy Bypass -File install.ps1", "installs ARM64 Python if missing · virtual environment · models for your chipset (one time) · espeak-ng · Start Menu shortcut"],
          Inches(0.7), Inches(1.6), Inches(12), Inches(1.1), fill=PANEL, color=ACCENT, size=18, bold=True)
    d.bullets(s, [
        "Runs on every Snapdragon X, X Plus, X Elite, X2 Plus and X2 Elite PC,| including HP OmniBook X, 5, 3, Ultra and EliteBook Ultra. 16 GB RAM recommended; --no-vlm on tighter machines.",
        "No admin rights, no account, no cloud.| Frames, audio and screenshots stay in RAM. Log file for support.",
        "Accessible by design:| global hotkeys, tray toggles, spoken confirmation of every action, a heads-up panel that shows what was heard and done, English and Hindi voices.",
        "Switch access:| a five-line JSON protocol turns any Arduino, including the UNO Q from the AI Lab kit, into a Sahaay switch. Try it in the Wokwi simulator without hardware.",
        "Reviewer tools:| tools/npu_check.py proves the NPU is active; tools/bench_all.py regenerates every number in this deck.",
    ], Inches(0.7), Inches(3.0), Inches(12), Inches(4), size=14, gap=10)

    # 9 Why on-device / HP
    s = d.slide("Why this belongs on a Snapdragon HP PC", kicker="USE CASE & INNOVATION")
    rows = [["Need", "Cloud assistant", "Sahaay on the NPU"],
            ["Camera on your face all day, screen with your bank statement", "uploaded frame by frame", "never leaves RAM"],
            ["Cursor latency a person can tolerate", "100 ms and up, network dependent", "under 1 ms per frame"],
            ["Hostel, hospital, train, patchy 4G", "does not work", "fully offline"],
            ["Battery on a laptop that is the user's only interface", "CPU or GPU inference drains it", "NPU runs the whole stack"],
            ["Cost to a student", "subscription + speech API minutes", "zero, forever"]]
    d.table(s, rows, Inches(0.6), Inches(1.6), Inches(8), [Inches(3.6), Inches(2.3), Inches(2.1)], size=12, row_h=Inches(0.5))
    d.bullets(s, [
        "What is new:| not a head mouse and not a screen reader, but one conversational layer over Windows that sees the screen the way the user needs, on the NPU, in Hindi and English.",
        "Fits the program:| HP India targets students with the OmniBook 3 and 5; Qualcomm India wants Snapdragon PCs under Rs 60,000. Sahaay makes that laptop an assistive device.",
        "Scales with silicon:| built within 45 TOPS and 16 GB; on the 80 TOPS X2 the same code runs bigger vision models and adds gaze.",
    ], Inches(8.9), Inches(1.6), Inches(4), Inches(5), size=12, gap=10)

    # 10 Development flow + hard parts
    s = d.slide("How it was built in three days, and what was hard", kicker="DEVELOPMENT FLOW")
    d.bullets(s, [
        "1. Toolchain proof:| ORT QNN on ARM64, AI Hub, GenieX; found the silent CPU fallback and fixed it in npu.py.",
        "2. Face:| AI Hub export, numpy MediaPipe pipeline, pose and blink metrics, 28 fps.",
        "3. Head cursor:| joystick mapping, dwell ring, blink and mouth gestures, calibration.",
        "4. Speech:| precompiled Whisper, KV-cache decode loop, VAD.",
        "5. Brain:| two-tier commands, Qwen3-4B tool schema, screen snapshot, validation.",
        "6. Narrator:| summaries, focused text, Qwen3-VL, Piper en/hi.",
        "7. App:| threads, overlay, tray, hotkeys, logging.  8. Deployment:| installer, model fetcher, benchmarks, Arduino bridge.",
    ], Inches(0.7), Inches(1.6), Inches(6.2), Inches(5.2), size=13, gap=7)
    d.text(s, "Easy", Inches(7.3), Inches(1.6), Inches(5.5), Inches(0.4), size=16, color=ACCENT, bold=True)
    d.text(s, "Qualcomm AI Hub: every model exported and profiled on real X2 Elite and X Elite devices in minutes, with per-op compute-unit reports. GenieX: the language and vision models were a pip install and one pull, on the NPU first try.",
           Inches(7.3), Inches(2.0), Inches(5.5), Inches(1.5), size=12, color=FG)
    d.text(s, "Hard", Inches(7.3), Inches(3.5), Inches(5.5), Inches(0.4), size=16, color=WARN, bold=True)
    d.text(s, "Windows ARM64 plumbing: no OpenCV wheel (DirectShow COM instead, CoInitialize on every thread); the Piper wheel's espeak bridge is not built for ARM64 (shell out to espeak-ng, drive VITS through ONNX Runtime); QNN context caches must be reused, not regenerated; two GenieX models cannot load at the same moment; and the QNN provider must be selected through the new device API or ONNX Runtime quietly runs on the CPU.",
           Inches(7.3), Inches(3.9), Inches(5.5), Inches(2.8), size=12, color=FG)

    # 11 Roadmap + close
    s = d.slide("Next, and thank you", kicker="ROADMAP")
    d.bullets(s, [
        "Gaze estimation (eyegaze on AI Hub)| for coarse screen-region selection alongside head pose.",
        "Piper on the NPU| via the AI Hub pipertts recipe; Hindi and Hinglish command grammar.",
        "Element grounding from the vision model:| \"click the blue button\" when there is no accessibility tree.",
        "OmniBook presence sensor| to pause tracking when the user steps away; MSIX package for the Microsoft Store.",
        "User trials| with an assistive-technology centre; calibration profiles for tremor and limited neck range.",
    ], Inches(0.7), Inches(1.6), Inches(7.5), Inches(4.5), size=15, gap=12)
    d.box(s, ["github.com/Vision-jarvis/sahaay", "code · benchmarks · install.ps1 · demo GIFs"], Inches(8.6), Inches(1.8), Inches(4.2), Inches(1.4), fill=PANEL, color=ACCENT, size=16, bold=True)
    d.box(s, ["Ruhan Srivastava", "IIT Kharagpur"], Inches(8.6), Inches(3.5), Inches(4.2), Inches(1.2), fill=PANEL, color=FG, size=16, bold=True)
    d.text(s, "Built entirely on a Snapdragon X2 Elite laptop with Qualcomm AI Hub, ONNX Runtime QNN and GenieX.", Inches(0.7), Inches(6.3), Inches(12), Inches(0.5), size=13, color=DIM)

    d.save(out)
    print(f"wrote {out} ({d.n} slides)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "deck" / "Sahaay_Pitch.pptx"))
    a = ap.parse_args()
    build(Path(a.out))
