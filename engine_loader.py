# engine_loader.py
# Local-only import guard + simple engine/device helpers (with back-compat wrappers)
from __future__ import annotations
import sys, os
from pathlib import Path

BASE = Path(__file__).parent.resolve()
PKGS_DIR = BASE / "_pkgs"

def _add_local_pkgs(strict: bool = True) -> None:
    """Prepend ./_pkgs and (if strict) strip global site/dist-packages."""
    p = str(PKGS_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    if strict:
        keep = []
        for sp in sys.path:
            low = sp.replace("\\", "/").lower()
            if "site-packages" in low or "dist-packages" in low:
                if Path(sp).resolve() == PKGS_DIR.resolve():
                    keep.append(sp)  # keep only our local _pkgs
                else:
                    continue
            else:
                keep.append(sp)
        sys.path[:] = keep

# Enforce local-first and strip globals even if start_app.py is launched directly
_add_local_pkgs(strict=True)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

# ----------------- Device helpers -----------------

def _has_cuda_pt() -> bool:
    try:
        import torch  # type: ignore
        return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        return False

def select_device_auto() -> str:
    """Return '0' if CUDA is available, otherwise 'cpu'."""
    return "0" if _has_cuda_pt() else "cpu"

def resolve_device(engine: str | None = "auto", device: str | None = "auto") -> str:
    """
    Back-compat helper expected by app_core.py.
    - device='auto' -> GPU if available, else CPU
    - device in {'cpu','-1'} -> 'cpu'
    - device in {'cuda','gpu','0','1',...} -> '0' (we run single-GPU by default)
    """
    dv = (device or "auto").strip().lower()
    if dv in ("cpu", "-1"):
        return "cpu"
    if dv in ("cuda", "gpu", "0", "1"):
        # We standardize to '0' (first CUDA device) for Ultralytics
        return "0"
    # auto
    return select_device_auto()

# ----------------- Model loader -----------------

def load_engine(model_path: str, engine: str | None = "auto"):
    """
    Return a Ultralytics YOLO model object for .pt or .onnx.
    Engine selection is mostly by file extension; 'engine' kept for compatibility.
    """
    from ultralytics import YOLO  # will import from ./_pkgs due to _add_local_pkgs
    ext = Path(model_path).suffix.lower()
    chosen = (engine or "auto").lower()
    if chosen == "auto":
        chosen = "onnx" if ext == ".onnx" else "pt"

    if chosen == "onnx":
        # Ensure onnxruntime exists in local _pkgs; otherwise show a clear message
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            print("[ERROR] engine=onnx but onnxruntime is not installed in ./_pkgs.")
            print("        Run bootstrap_env.py again or:")
            print("        python -m pip install --target _pkgs onnxruntime  (or onnxruntime-gpu)")
            raise
        if ext != ".onnx":
            print("[WARN] engine=onnx chosen but file is not .onnx; continuing…")

    # For .pt models Ultralytics will use torch; for .onnx it prefers ORT under the hood.
    return YOLO(model_path)

# ----------------- Back-compat wrappers -----------------

def load_yolo_model(model_path: str, engine: str | None = "auto"):
    """Alias kept for older code (app_core.py expects this name)."""
    return load_engine(model_path, engine=engine)
