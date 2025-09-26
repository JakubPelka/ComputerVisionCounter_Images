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
    Loads a YOLO model with best-effort ONNX support.

    - .pt   -> Ultralytics YOLO (PyTorch)
    - .onnx -> Ultralytics ONNXRuntime wrapper
              * For many non-Ultralytics exports (e.g., YOLOv7), metadata like
                'stride' or 'task' may be missing. We inject safe defaults so
                Ultralytics' preprocessing (letterbox) doesn't crash.

    Notes:
      - We do NOT change your inference calls; this only makes the model object
        resilient when metadata is absent.
    """
    from ultralytics import YOLO  # type: ignore

    ext = _p.splitext(model_path)[1].lower()
    chosen = (engine or "auto").lower()
    if chosen == "auto":
        chosen = "onnx" if ext == ".onnx" else "pt"

    # Sanity: ensure ORT present for .onnx paths
    if chosen == "onnx":
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            print("[ERROR] engine=onnx but onnxruntime is not installed.")
            print("        Re-run bootstrap or: pip install onnxruntime (or onnxruntime-gpu)")
            sys.exit(2)
        if ext != ".onnx":
            print("[WARN] engine=onnx chosen but file is not .onnx; continuing…")

    # Let Ultralytics select backend by extension
    m = YOLO(model_path)

    # ---- ONNX-specific resilience (stride + task + imgsz) -------------------
    if ext == ".onnx":
        # 1) Add a default 'stride' if missing (common for YOLOv7 ONNX)
        def _has_stride(obj) -> bool:
            try:
                s = getattr(obj, "stride", None)
                return s is not None
            except Exception:
                return False

        stride_ok = False
        for obj in (getattr(m, "model", None),
                    getattr(m, "predictor", None),
                    getattr(getattr(m, "predictor", None), "model", None)):
            if obj and _has_stride(obj):
                stride_ok = True
                break

        if not stride_ok:
            # inject stride=32 on both potential holders
            for obj in (getattr(m, "model", None),
                        getattr(getattr(m, "predictor", None), "model", None)):
                try:
                    if obj is not None:
                        setattr(obj, "stride", 32)
                except Exception:
                    pass

        # 2) Quiet the "Unable to automatically guess model task" by setting detect
        try:
            if hasattr(m, "overrides"):
                m.overrides["task"] = m.overrides.get("task", "detect") or "detect"
        except Exception:
            pass
        # Some versions use predictor.args; keep both safe
        try:
            if hasattr(m, "predictor") and hasattr(m.predictor, "args"):
                args = m.predictor.args
                if hasattr(args, "task") and (args.task in (None, "", "predict")):
                    args.task = "detect"
        except Exception:
            pass

        # 3) Ensure a sane imgsz if it wasn't set (helps letterboxing)
        try:
            if hasattr(m, "overrides"):
                m.overrides["imgsz"] = max(640, int(m.overrides.get("imgsz", 640)))
        except Exception:
            pass

    return m
