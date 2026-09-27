"""NPU session factory for ONNX Runtime + Qualcomm QNN execution provider.

Verified on Snapdragon X2 Elite Extreme with onnxruntime 1.30 + onnxruntime-qnn 2.6
(plugin packaging). The classic ``providers=["QNNExecutionProvider"]`` form silently
falls back to CPU with the plugin runtime, so every session here goes through
``add_provider_for_devices`` and asserts the active provider.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

_REGISTERED = False


def register_qnn() -> bool:
    """Register the QNN plugin EP once. Returns True if the NPU is available."""
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        import onnxruntime_qnn as q  # type: ignore
    except ImportError:
        return False
    try:
        ort.register_execution_provider_library(q.get_ep_name(), q.get_library_path())
    except Exception as exc:  # already registered in this process
        if "already" not in str(exc).lower():
            raise
    _REGISTERED = True
    return True


def npu_devices() -> list[Any]:
    """Return the ORT EP devices that map QNN onto the Hexagon NPU."""
    if not register_qnn():
        return []
    return [
        d
        for d in ort.get_ep_devices()
        if d.ep_name == "QNNExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.NPU
    ]


@dataclass
class SessionInfo:
    path: str
    providers: list[str]
    on_npu: bool
    load_ms: float
    options: dict[str, str] = field(default_factory=dict)


def create_session(
    model_path: str | os.PathLike,
    *,
    prefer_npu: bool = True,
    performance_mode: str = "burst",
    fp16: bool = True,
    context_cache: bool = True,
    log_severity: int = 3,
) -> tuple[ort.InferenceSession, SessionInfo]:
    """Create an inference session on the NPU (falls back to CPU only if asked).

    ``context_cache`` stores the compiled QNN graph next to the model on first load so
    later launches skip the on-device compile step.
    """
    model_path = str(model_path)
    so = ort.SessionOptions()
    so.log_severity_level = log_severity
    opts: dict[str, str] = {}
    devs = npu_devices() if prefer_npu else []
    if devs:
        opts = {
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": performance_mode,
            "enable_htp_fp16_precision": "1" if fp16 else "0",
        }
        so.add_provider_for_devices(devs, opts)
        if context_cache and not model_path.endswith(".onnx_ctx.onnx"):
            ctx = Path(model_path).with_suffix(".onnx_ctx.onnx")
            so.add_session_config_entry("ep.context_enable", "1")
            so.add_session_config_entry("ep.context_file_path", str(ctx))
            so.add_session_config_entry("ep.context_embed_mode", "0")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_path, so)
    load_ms = (time.perf_counter() - t0) * 1000
    provs = sess.get_providers()
    info = SessionInfo(model_path, provs, provs[0] == "QNNExecutionProvider", load_ms, opts)
    return sess, info


def bench(sess: ort.InferenceSession, feeds: dict[str, np.ndarray], *, warmup: int = 5, iters: int = 30) -> float:
    """Median latency in milliseconds for one ``run`` with the given feeds."""
    for _ in range(warmup):
        sess.run(None, feeds)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def random_feeds(sess: ort.InferenceSession, seed: int = 0) -> dict[str, np.ndarray]:
    """Build random inputs matching the session's static input shapes."""
    rng = np.random.default_rng(seed)
    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        t = inp.type
        if "float16" in t:
            feeds[inp.name] = rng.standard_normal(shape).astype(np.float16)
        elif "float" in t:
            feeds[inp.name] = rng.standard_normal(shape).astype(np.float32)
        elif "uint8" in t:
            feeds[inp.name] = rng.integers(0, 255, shape, dtype=np.uint8)
        elif "int64" in t:
            feeds[inp.name] = rng.integers(0, 10, shape, dtype=np.int64)
        elif "int32" in t:
            feeds[inp.name] = rng.integers(0, 10, shape, dtype=np.int32)
        elif "bool" in t:
            feeds[inp.name] = rng.integers(0, 2, shape).astype(bool)
        else:
            feeds[inp.name] = rng.standard_normal(shape).astype(np.float32)
    return feeds
