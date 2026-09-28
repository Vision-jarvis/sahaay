# Qualcomm AI Hub profiling on reference devices

Every ONNX model Sahaay runs through ONNX Runtime QNN was compiled and profiled in Qualcomm AI Hub's device farm on two targets: the **Snapdragon X2 Elite CRD** (Hexagon v81, the class of the development laptop) and the **Snapdragon X Elite CRD** (Hexagon v73, the chip in the HP OmniBook X and EliteBook Ultra; X Plus and X share the same NPU generation). Job pages are public and show the per-op compute-unit assignment.

| Model | Snapdragon X2 Elite CRD | Snapdragon X Elite CRD | Ops on NPU | Peak memory |
|---|---:|---:|---|---|
| Face detector (BlazeFace, 256x256) | 0.4 ms ([jgly9mxe5](https://workbench.aihub.qualcomm.com/jobs/jgly9mxe5/)) | 0.7 ms ([jgnzdm9kg](https://workbench.aihub.qualcomm.com/jobs/jgnzdm9kg/)) | 145 / 145 | 1 MB |
| Face landmark detector (468 points, 192x192) | 0.2 ms ([j568947vg](https://workbench.aihub.qualcomm.com/jobs/j568947vg/)) | 0.3 ms ([jprlm240p](https://workbench.aihub.qualcomm.com/jobs/jprlm240p/)) | 105 / 105 | 1 MB |
| Whisper base encoder (30 s window) | 21.5 ms ([jg9z7n2qp](https://workbench.aihub.qualcomm.com/jobs/jg9z7n2qp/)) | 45.4 ms ([jgk2wqqwg](https://workbench.aihub.qualcomm.com/jobs/jgk2wqqwg/)) | 556 / 556 | 66 MB |
| Whisper base decoder (one token) | 2.4 ms ([jp1nkz1kg](https://workbench.aihub.qualcomm.com/jobs/jp1nkz1kg/)) | 3.7 ms ([j5qlxrrnp](https://workbench.aihub.qualcomm.com/jobs/j5qlxrrnp/)) | 975 / 975 | 20 to 125 MB |
| FaceMap 3DMM (128x128, alternative pose model) | 0.2 ms ([jgol76r4g](https://workbench.aihub.qualcomm.com/jobs/jgol76r4g/)) | not profiled | 56 / 56 | under 1 MB |

Tool versions on the farm: QAIRT 2.50, ONNX Runtime 1.27.1. Precision float (fp16 on HTP).

What this means for a judge's OmniBook (X Elite): face tracking costs about 1 ms per frame, a six-second command costs roughly 45 ms of encoder plus 3.7 ms per output token, so about 120 ms end to end before the language model. Every op of every model runs on the NPU; nothing falls back to the CPU or GPU.

Reproduce: `set QAIHM_CI=1` then `python -m qai_hub_models.models.<model>.export --device "Snapdragon X Elite CRD" --target-runtime onnx --skip-inferencing` (use `precompiled_qnn_onnx` for `whisper_base`). Requires an AI Hub token and x64 Python for `qai_hub_models`.
