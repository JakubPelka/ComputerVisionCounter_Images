# engine_loader.py
from __future__ import annotations
import sys, os
import os.path as _p

def _add_local_pkgs():
    here = _p.dirname(_p.abspath(__file__))
    for d in ["_pkgs", "pkgs"]:
        cand = _p.join(here, d)
        if _p.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
_add_local_pkgs()

def _has_cuda_pt() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False

def _has_ort_cuda() -> bool:
    try:
        import onnxruntime as ort  # type: ignore
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False

def resolve_device(engine: str, device_arg: str = "auto"):
    """
    Returns '0' (GPU) or 'cpu':
      - If device_arg != 'auto' -> return it as-is.
      - ONNX: if CUDAExecutionProvider available -> 0 else 'cpu'
      - PT/auto: if torch.cuda.is_available() -> 0 else 'cpu'
    """
    if device_arg and str(device_arg).lower() != "auto":
        return device_arg
    eng = (engine or "auto").lower()
    if eng == "onnx":
        return 0 if _has_ort_cuda() else "cpu"
    return 0 if _has_cuda_pt() else "cpu"

def load_yolo_model(model_path: str, engine: str = "auto"):
    """
    Minimal wrapper: Ultralytics YOLO supports .pt and .onnx (with onnxruntime).
    The engine flag is used for sanity checks and messaging; YOLO decides backend by file extension.
    """
    from ultralytics import YOLO  # type: ignore

    ext = _p.splitext(model_path)[1].lower()
    chosen = (engine or "auto").lower()
    if chosen == "auto":
        chosen = "onnx" if ext == ".onnx" else "pt"

    if chosen == "onnx":
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            print("[ERROR] engine=onnx but onnxruntime is not installed.")
            print("        Re-run bootstrap or: pip install onnxruntime (or onnxruntime-gpu)")
            sys.exit(2)
        if ext != ".onnx":
            print("[WARN] engine=onnx chosen but file is not .onnx; continuing…")

    return YOLO(model_path)