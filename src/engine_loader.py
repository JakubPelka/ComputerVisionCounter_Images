# engine_loader.py
# Local-only import guard + engine/device helpers, with *automatic* ONNX metadata patching.
from __future__ import annotations
import sys, os, json, ast, re
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
    - device in {'cuda','gpu','0','1',...} -> '0' (single-GPU default)
    """
    dv = (device or "auto").strip().lower()
    if dv in ("cpu", "-1"):
        return "cpu"
    if dv in ("cuda", "gpu", "0", "1"):
        return "0"
    return select_device_auto()

# ----------------- Ultralytics auto-install blocker -----------------

def _disable_ultralytics_auto_pip() -> None:
    """
    Ultralytics may try to 'check_requirements' (pip into system). We block that to
    keep everything local.
    """
    try:
        import ultralytics  # noqa: F401
        from ultralytics.utils import checks  # type: ignore
        def _noinstall(*args, **kwargs):
            return True
        checks.check_requirements = _noinstall  # type: ignore
        os.environ.setdefault("YOLO_VERBOSE", "0")
    except Exception:
        pass

# ----------------- ONNX metadata helpers -----------------

def _read_onnx_metadata(model_path: str) -> dict:
    """Return metadata dict from ONNX (keys as strings)."""
    try:
        import onnx  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "ONNX Python package is not available in ./_pkgs. "
            "Run bootstrap_env.py again."
        ) from e
    m = onnx.load(model_path)
    try:
        meta = {p.key: p.value for p in getattr(m, "metadata_props", [])}
    except Exception:
        meta = {}
    return meta

def _parse_names(meta: dict) -> list[str] | None:
    raw = meta.get("names") or meta.get("classes") or meta.get("class_names")
    if not raw:
        return None
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            parsed = None
    if isinstance(parsed, dict):
        try:
            keys = sorted(parsed.keys(), key=lambda k: int(k))
            return [str(parsed[k]) for k in keys]
        except Exception:
            return [str(v) for v in parsed.values()]
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return None

def _infer_default_names_from_filename(path: Path) -> list[str]:
    # e.g., "car_aerial_detection_yolo7.onnx" -> ["car"]
    m = re.match(r"([a-zA-Z0-9]+)", path.stem)
    token = (m.group(1).lower() if m else "object")
    # Very naive plural → singular heuristics could be added; keep simple
    return [token]

def _write_ultra_metadata(in_path: str, out_path: str, names: list[str], stride: int) -> None:
    """Write 'names' dict, 'stride', 'task=detect', plus hints 'batch','imgsz','nc'."""
    import onnx  # type: ignore
    m = onnx.load(in_path)

    def _set_meta(model, key: str, value: str) -> None:
        for p in model.metadata_props:
            if p.key == key:
                p.value = value
                return
        q = model.metadata_props.add()
        q.key = key
        q.value = value

    names_dict = {i: str(n) for i, n in enumerate(names)}
    _set_meta(m, "names", json.dumps(names_dict))
    _set_meta(m, "stride", str(int(stride)))
    _set_meta(m, "task", "detect")
    _set_meta(m, "batch", "1")
    _set_meta(m, "imgsz", "640")  # hint only; app may override/tile anyway
    _set_meta(m, "nc", str(len(names)))

    onnx.save(m, out_path)

def _ensure_ultra_metadata(model_path: str) -> tuple[str, list[str], int]:
    """
    Ensure ONNX has the minimum Ultralytics metadata.
    If missing/partial, write a patched copy alongside the original and return its path.
    Returns: (path_to_use, names, stride)
    """
    src = Path(model_path)
    meta = _read_onnx_metadata(str(src))
    names = _parse_names(meta)
    stride = None
    try:
        if "stride" in meta:
            stride = int(meta["stride"])
    except Exception:
        stride = None
    task = meta.get("task")
    batch = meta.get("batch")

    # If everything looks fine, use as-is
    if names and stride and task == "detect" and batch:
        return str(src), names, stride

    # Need to patch: pick defaults
    if not names:
        names = _infer_default_names_from_filename(src)
    if not stride:
        stride = 32

    # Create unique patched filename
    outp = src.with_name(src.stem + "_ultra.onnx")
    if outp.exists():
        i = 2
        while True:
            cand = src.with_name(f"{src.stem}_ultra_{i}.onnx")
            if not cand.exists():
                outp = cand
                break
            i += 1

    _write_ultra_metadata(str(src), str(outp), names, stride)
    print(f"[INFO] Patched ONNX → {outp.name} (classes={len(names)}, stride={stride}, task=detect)")
    return str(outp), names, stride

# ----------------- Model loader -----------------

def load_engine(model_path: str, engine: str | None = "auto"):
    """
    Return a Ultralytics YOLO model object for .pt or .onnx.
    For .onnx, metadata is auto-patched (no GUI prompts).
    """
    ext = Path(model_path).suffix.lower()
    chosen = (engine or "auto").lower()
    if chosen == "auto":
        chosen = "onnx" if ext == ".onnx" else "pt"

    if chosen == "onnx":
        # Ensure both ORT and ONNX exist locally so Ultralytics won't try system pip
        try:
            import onnxruntime  # noqa: F401
            import onnx  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "ONNX / ONNXRuntime not found in ./_pkgs. Run bootstrap_env.py again."
            ) from e

        # Ensure metadata (names/stride/task/batch) — auto-patch if needed
        model_path, names_list, stride = _ensure_ultra_metadata(model_path)

        # Now load with Ultralytics
        _disable_ultralytics_auto_pip()
        from ultralytics import YOLO
        m = YOLO(model_path, task="detect")

        # Best-effort: ensure names/stride are visible on the model object for the UI
        try:
            if not getattr(m, "names", None) and names_list:
                m.names = {i: n for i, n in enumerate(names_list)}
        except Exception:
            pass
        try:
            mdl = getattr(m, "model", None)
            if mdl is not None:
                if not getattr(mdl, "names", None) and names_list:
                    mdl.names = {i: n for i, n in enumerate(names_list)}
                if not getattr(mdl, "stride", None) and stride:
                    mdl.stride = stride
        except Exception:
            pass

        return m

    # .pt path (unchanged)
    _disable_ultralytics_auto_pip()
    from ultralytics import YOLO
    return YOLO(model_path)

# ----------------- Back-compat wrappers -----------------

def load_yolo_model(model_path: str, engine: str | None = "auto"):
    """Alias kept for older code (app_core.py expects this name)."""
    return load_engine(model_path, engine=engine)
