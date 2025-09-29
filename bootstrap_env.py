# bootstrap_env.py
# Bootstrapper for ComputerVision Counter
# - Installs only missing/outdated packages into ./_pkgs (no forced reinstall)
# - Enforces local-only imports (strip global site/dist-packages)
# - Prints module versions and file paths (so you can verify they load from ./_pkgs)
# - Sets PYTHONNOUSERSITE=1
# - Launches your app (default: start_app.py; override with arg)

from __future__ import annotations
import os, sys, subprocess, socket, runpy, re
from pathlib import Path
from importlib import import_module
from importlib.metadata import distributions, PackageNotFoundError, version as get_version

BASE = Path(__file__).parent.resolve()
PKGS_DIR = BASE / "_pkgs"
DEFAULT_ENTRY = "start_app.py"

# ---------- logging / net ----------
def log(msg: str) -> None:
    print(msg, flush=True)

def online() -> bool:
    try:
        socket.create_connection(("pypi.org", 443), timeout=3).close()
        return True
    except OSError:
        return False

# ---------- sys.path controls ----------
def add_local_pkgs_and_strip_globals() -> None:
    """Prepend ./_pkgs and strip any other site-packages/dist-packages entries."""
    p = str(PKGS_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)

    keep: list[str] = []
    for sp in sys.path:
        low = sp.replace("\\", "/").lower()
        if "site-packages" in low or "dist-packages" in low:
            # keep only our local _pkgs entry
            if Path(sp).resolve() == PKGS_DIR.resolve():
                keep.append(sp)
            else:
                continue
        else:
            keep.append(sp)
    sys.path[:] = keep

# ---------- versions present in _pkgs ----------
def _local_versions() -> dict[str, str]:
    """Return {normalized-name: version} for distributions INSIDE ./_pkgs only."""
    vers: dict[str, str] = {}
    try:
        for dist in distributions(path=[str(PKGS_DIR)]):
            name = (dist.metadata.get("Name") or "").strip()
            if not name:
                continue
            vers[name.lower()] = dist.version
    except Exception:
        pass
    return vers

def _norm_version_tuple(v: str) -> tuple[int, ...]:
    parts = [int(x) for x in re.findall(r"\d+", v)]
    return tuple(parts) if parts else (0,)

def _satisfies(installed: str, op: str, required: str) -> bool:
    a, b = _norm_version_tuple(installed), _norm_version_tuple(required)
    if op == "==":
        return a == b
    # default to '>='
    return a >= b

def _parse_req(req: str) -> tuple[str, str, str]:
    """Parse 'name==x.y' or 'name>=x.y' → (name, op, ver)."""
    if "==" in req:
        name, ver = req.split("==", 1)
        return name.strip(), "==", ver.strip()
    if ">=" in req:
        name, ver = req.split(">=", 1)
        return name.strip(), ">=", ver.strip()
    # no spec → treat as >=0
    return req.strip(), ">=", "0"

# ---------- pip ----------
def run_pip(args: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    cmd = [sys.executable, "-m", "pip"] + args
    log(f"[pip] {' '.join(args)}")
    return subprocess.call(cmd, env=env)

def ensure_pkg(req: str) -> bool:
    """
    Ensure requirement 'name==x' or 'name>=x' is present in ./_pkgs.
    Returns True if satisfied (already or after install), False on failure.
    """
    name, op, ver = _parse_req(req)
    present = _local_versions()
    inst = present.get(name.lower())

    if inst and _satisfies(inst, op, ver):
        log(f"[ok] {name} {inst} (already in _pkgs)")
        return True

    # not present or too old → install into _pkgs
    rc = run_pip(["install", "--no-cache-dir", "--no-warn-script-location",
                  "--target", str(PKGS_DIR), req])
    if rc != 0:
        log(f"[ERR] pip failed for {req}")
        return False

    # re-check
    present = _local_versions()
    inst = present.get(name.lower())
    if inst and _satisfies(inst, op, ver):
        log(f"[ok] {name} {inst} (installed to _pkgs)")
        return True

    log(f"[ERR] {name} did not install or wrong version after pip")
    return False

# ---------- install set ----------
def install_minimal_online() -> None:
    """
    Only install if missing or too old. No forced upgrades on subsequent runs.
    """
    # IMPORTANT: accept newer NumPy too to avoid downgrade loops
    ensure_pkg("numpy>=1.26.4")

    ensure_pkg("opencv-python>=4.8.0")
    ensure_pkg("ultralytics>=8.3.0")
    ensure_pkg("shapely>=2.0.0")
    ensure_pkg("pyproj>=3.6.0")
    ensure_pkg("onnx>=1.14.0")

    # ONNX Runtime: install only if neither CPU nor GPU build present
    present = _local_versions()
    has_ort = ("onnxruntime" in present) or ("onnxruntime-gpu" in present)
    if not has_ort:
        rc = run_pip(["install", "--no-cache-dir", "--no-warn-script-location",
                      "--target", str(PKGS_DIR), "onnxruntime-gpu"])
        if rc != 0:
            log("[INFO] onnxruntime-gpu failed or not suitable; trying CPU build…")
            run_pip(["install", "--no-cache-dir", "--no-warn-script-location",
                     "--target", str(PKGS_DIR), "onnxruntime"])

# ---------- banner ----------
def smoke_imports_print_paths() -> None:
    def info(modname: str):
        try:
            m = import_module(modname)
            ver = getattr(m, "__version__", None)
            try:
                ver = ver or get_version(modname)
            except PackageNotFoundError:
                pass
            path = getattr(m, "__file__", "?")
            print(f"[OK] import {modname} ({ver}) @ {path}")
        except Exception as e:
            print(f"[MISS] {modname} — {e}")

    for mod in ["numpy", "cv2", "ultralytics", "torch", "torchvision", "onnxruntime"]:
        info(mod)

def verify_yolo11_shapes() -> None:
    try:
        from ultralytics.nn.modules import C3k2  # type: ignore
        print("[OK] YOLOv11 modules present (C3k2).")
    except Exception:
        print("[WARN] YOLOv11 modules not detected. If your model needs them, upgrade ultralytics.")
        print("       Try: pip install --upgrade --target _pkgs ultralytics")

# ---------- main ----------
def main():
    os.environ["PYTHONNOUSERSITE"] = "1"

    PKGS_DIR.mkdir(parents=True, exist_ok=True)
    add_local_pkgs_and_strip_globals()

    print("=== ComputerVision Counter — Bootstrap ===")
    print(f"Base folder:   {BASE}")
    print(f"Local pkgs:    {PKGS_DIR}")

    if online():
        print("[NET] Online → ensuring packages in ./_pkgs (skip if already satisfied)…")
        install_minimal_online()
    else:
        print("[NET] Offline — using existing packages in ./_pkgs")

    add_local_pkgs_and_strip_globals()

    print("\n=== Import check (version + path) ===")
    smoke_imports_print_paths()
    verify_yolo11_shapes()
    print("=====================================\n")

    entry = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ENTRY
    entry_path = (BASE / entry)
    if not entry_path.exists():
        print(f"[ERROR] Entry script not found: {entry_path}")
        sys.exit(2)

    print(f"[RUN] {entry_path}")
    runpy.run_path(str(entry_path), run_name="__main__")

if __name__ == "__main__":
    main()
