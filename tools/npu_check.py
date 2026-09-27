"""Prove that models run on the Hexagon NPU, and by how much it matters.

Usage:
    py -3.12 tools/npu_check.py                      # environment + built-in conv test
    py -3.12 tools/npu_check.py models/*.onnx        # also benchmark real models
    py -3.12 tools/npu_check.py --json out.json ...  # machine-readable report

For every model it creates two sessions (CPU only, NPU via QNN), asserts which
execution provider is actually active, and reports median latency. The output is
committed to benchmarks/ so reviewers can see measured, not claimed, numbers.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sahaay.npu import bench, create_session, npu_devices, random_feeds, register_qnn  # noqa: E402


def env_report() -> dict:
    rep = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "onnxruntime": ort.__version__,
        "providers_available": ort.get_available_providers(),
        "qnn_plugin": None,
        "ep_devices": [],
    }
    if register_qnn():
        import onnxruntime_qnn as q  # type: ignore

        rep["qnn_plugin"] = {"library": q.get_library_path(), "ep": q.get_ep_name()}
        rep["providers_available"] = ort.get_available_providers()
        for d in ort.get_ep_devices():
            rep["ep_devices"].append(
                {"ep": d.ep_name, "vendor": d.ep_vendor, "device_type": str(d.device.type).split(".")[-1]}
            )
    rep["npu_devices_found"] = len(npu_devices())
    return rep


def builtin_conv_model(path: Path) -> Path:
    """A small 2-layer conv net so the check works without any downloaded model."""
    import onnx
    from onnx import TensorProto, helper

    rng = np.random.default_rng(0)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64, 1, 1])
    W1 = helper.make_tensor("W1", TensorProto.FLOAT, [32, 3, 3, 3], (rng.standard_normal((32, 3, 3, 3)) * 0.1).astype(np.float32).flatten())
    W2 = helper.make_tensor("W2", TensorProto.FLOAT, [64, 32, 3, 3], (rng.standard_normal((64, 32, 3, 3)) * 0.05).astype(np.float32).flatten())
    nodes = [
        helper.make_node("Conv", ["X", "W1"], ["c1"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("Conv", ["r1", "W2"], ["c2"], pads=[1, 1, 1, 1], strides=[2, 2]),
        helper.make_node("Relu", ["c2"], ["r2"]),
        helper.make_node("GlobalAveragePool", ["r2"], ["Y"]),
    ]
    g = helper.make_graph(nodes, "tiny", [X], [Y], initializer=[W1, W2])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(path))
    return path


def check_model(path: Path, iters: int) -> dict:
    row = {"model": path.name, "path": str(path)}
    cpu, cinfo = create_session(path, prefer_npu=False, context_cache=False)
    feeds = random_feeds(cpu)
    row["cpu_ms"] = round(bench(cpu, feeds, iters=iters), 3)
    row["cpu_providers"] = cinfo.providers
    y_cpu = cpu.run(None, feeds)
    del cpu
    try:
        npu, ninfo = create_session(path, prefer_npu=True)
        row["npu_ms"] = round(bench(npu, feeds, iters=iters), 3)
        row["npu_providers"] = ninfo.providers
        row["npu_active"] = ninfo.on_npu
        row["npu_load_ms"] = round(ninfo.load_ms, 1)
        y_npu = npu.run(None, feeds)
        diffs = [float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max()) for a, b in zip(y_cpu, y_npu) if a.dtype.kind == "f"]
        row["max_abs_diff"] = max(diffs) if diffs else None
        row["speedup"] = round(row["cpu_ms"] / row["npu_ms"], 2) if row["npu_ms"] else None
    except Exception as exc:
        row["npu_error"] = str(exc)[:400]
        row["npu_active"] = False
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="ONNX files to benchmark on CPU and NPU")
    ap.add_argument("--json", help="write full report here")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--no-builtin", action="store_true", help="skip the built-in conv test")
    args = ap.parse_args()

    rep = {"env": env_report(), "models": []}
    print("== Environment ==")
    for k in ("machine", "python", "onnxruntime", "providers_available", "npu_devices_found"):
        print(f"  {k}: {rep['env'][k]}")
    for d in rep["env"]["ep_devices"]:
        print(f"  ep device: {d['ep']} on {d['device_type']} ({d['vendor']})")

    paths = [Path(p) for p in args.models]
    if not args.no_builtin:
        paths.insert(0, builtin_conv_model(Path(__file__).resolve().parent / "_tiny_conv.onnx"))

    print("\n== Models ==")
    print(f"  {'model':40} {'cpu ms':>9} {'npu ms':>9} {'speedup':>8}  active")
    for p in paths:
        row = check_model(p, args.iters)
        rep["models"].append(row)
        npu_ms = row.get("npu_ms")
        print(
            f"  {row['model'][:40]:40} {row['cpu_ms']:9.3f} "
            f"{(npu_ms if npu_ms is not None else float('nan')):9.3f} "
            f"{(row.get('speedup') or 0):8.2f}  {'NPU' if row.get('npu_active') else 'CPU FALLBACK'}"
            + (f"  ({row['npu_error'][:80]})" if row.get("npu_error") else "")
        )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rep, indent=2))
        print(f"\nreport written to {args.json}")
    ok = rep["env"]["npu_devices_found"] > 0 and all(m.get("npu_active") for m in rep["models"])
    print("\nRESULT:", "all models executed on the Hexagon NPU" if ok else "NPU NOT ACTIVE for at least one model")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
