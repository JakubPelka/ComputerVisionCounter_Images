# bootstrap_env.py
# Windows-only bootstrapper: installs/updates packages to ./_pkgs
# 1) Try ONLINE (PyPI + official PyTorch CPU index)
# 2) If offline/failure -> OFFLINE from ./wheels
# 3) Prefer ONNX Runtime GPU; if it fails, fallback to CPU
# 4) Verify YOLOv11 support (C3k2); if missing -> force (re)install ultralytics
# 5) Run target app (default: start_app.py; override: pass filename as arg)

from __future__ import annotations
import sys, subprocess, os, socket, runpy
from pathlib import Path
from importlib import import_module
from importlib.metadata import version as get_version, PackageNotFoundError

BASE = Path(__file__).parent.resolve()
PKGS_DIR = BASE / "_pkgs"
WHEELS_DIR = BASE / "wheels"

# === target app ===
DEFAULT_APP = "start_app.py"
APP_PATH = BASE / (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_APP)

# === Requirements ===
# Torch/vision from CPU index – tested on Py 3.11/3.12 (CPU)
REQUIREMENTS_TORCH = [
    "torch==2.3.1+cpu",
    "torchvision==0.18.1+cpu",
]
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# Rest from PyPI
# NOTE: we install onnxruntime separately (prefer GPU), see below.
REQUIREMENTS_REST = [
    "ultralytics>=8.3.0",      # YOLOv11 (C3k2 module)
    "opencv-python>=4.8.0",
    "numpy>=1.26.0,<2.0.0",
    "pandas>=2.1.0",
    "pillow>=10.2.0",
    "tqdm>=4.66.0",
    "supervision>=0.21.0",
    "PyYAML>=6.0",
    "scipy>=1.11.0",
    "onnx>=1.15.0",
    "tifffile>=2023.4.12",     # GeoTIFF support for GeoJSON export
]

SMOKE_IMPORTS = [
    ("torch", None),
    ("torchvision", None),
    ("ultralytics", None),
    ("cv2", None),
    ("numpy", None),
    ("pandas", None),
    ("PIL", None),
    ("tqdm", None),
    ("yaml", None),
    ("scipy", None),
    ("onnxruntime", None),  # check ORT presence
    ("tifffile", None),
]

def is_online(host="pypi.org", port=443, timeout=2.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def add_pkgs_to_syspath():
    if str(PKGS_DIR) not in sys.path:
        sys.path.insert(0, str(PKGS_DIR))

def run_pip(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pip"] + args
    print(">>", " ".join(cmd))
    return subprocess.call(cmd)

def have_all_imports(verbose=True) -> bool:
    add_pkgs_to_syspath()
    ok = True
    for mod, _ in SMOKE_IMPORTS:
        try:
            import_module(mod)
            if verbose:
                try:
                    print(f"[OK] import {mod} ({get_version(mod)})")
                except PackageNotFoundError:
                    print(f"[OK] import {mod}")
        except Exception as e:
            ok = False
            if verbose:
                print(f"[MISS] {mod}: {e}")
    return ok

def ultralytics_supports_yolo11() -> bool:
    """YOLOv11 uses C3k2 in ultralytics.nn.modules.block."""
    try:
        add_pkgs_to_syspath()
        from ultralytics.nn.modules import block as ublock
        return hasattr(ublock, "C3k2")
    except Exception:
        return False

def install_onnxruntime_prefer_gpu_online() -> None:
    """Try onnxruntime-gpu first; fall back to CPU."""
    print("\n=== ONNX Runtime (prefer GPU) — ONLINE ===")
    rc_gpu = run_pip(["install", "--target", str(PKGS_DIR), "--upgrade", "onnxruntime-gpu"])
    if rc_gpu == 0:
        print("[OK] onnxruntime-gpu installed.")
        return
    print("[WARN] onnxruntime-gpu failed, falling back to CPU…")
    rc_cpu = run_pip(["install", "--target", str(PKGS_DIR), "--upgrade", "onnxruntime"])
    if rc_cpu != 0:
        print("[ERR] onnxruntime (CPU) install failed as well.")

def install_onnxruntime_prefer_gpu_offline() -> None:
    """OFFLINE: prefer onnxruntime-gpu*.whl; else onnxruntime*.whl."""
    print("\n=== ONNX Runtime (prefer GPU) — OFFLINE ===")
    if not WHEELS_DIR.exists():
        print("[ERR] ./wheels directory not found for offline ORT.")
        return
    whl_gpu = sorted(WHEELS_DIR.glob("onnxruntime_gpu*.whl")) + sorted(WHEELS_DIR.glob("onnxruntime-gpu*.whl"))
    whl_cpu = sorted(WHEELS_DIR.glob("onnxruntime*.whl"))

    if whl_gpu:
        rc = run_pip(["install", "--no-index", "--find-links", str(WHEELS_DIR),
                      "--target", str(PKGS_DIR), str(whl_gpu[0])])
        if rc == 0:
            print(f"[OK] Installed {whl_gpu[0].name}")
            return
        print("[WARN] onnxruntime-gpu wheel failed, trying CPU…")

    # CPU fallback; filter out gpu wheels
    whl_cpu = [p for p in whl_cpu if "gpu" not in p.name.lower()]
    if whl_cpu:
        rc2 = run_pip(["install", "--no-index", "--find-links", str(WHEELS_DIR),
                       "--target", str(PKGS_DIR), str(whl_cpu[0])])
        if rc2 == 0:
            print(f"[OK] Installed {whl_cpu[0].name}")
            return
    print("[ERR] No suitable onnxruntime wheel installed offline.")

def install_online() -> bool:
    print("\n=== ONLINE INSTALL (to ./_pkgs) ===")
    PKGS_DIR.mkdir(parents=True, exist_ok=True)

    run_pip(["install", "--upgrade", "pip"])

    rc_t = run_pip([
        "install", "--index-url", PYTORCH_CPU_INDEX, "--target", str(PKGS_DIR),
        *REQUIREMENTS_TORCH
    ])
    if rc_t != 0:
        print("[WARN] Torch CPU install failed (continuing).")

    rc_r = run_pip(["install", "--target", str(PKGS_DIR), *REQUIREMENTS_REST])
    if rc_r != 0:
        print("[WARN] Some packages failed from PyPI.")

    # ONNX Runtime: prefer GPU, fallback CPU
    install_onnxruntime_prefer_gpu_online()

    return have_all_imports(verbose=True)

def install_offline_from_wheels() -> bool:
    print("\n=== OFFLINE INSTALL from ./wheels (to ./_pkgs) ===")
    if not WHEELS_DIR.exists():
        print("[ERR] ./wheels directory not found.")
        return False

    PKGS_DIR.mkdir(parents=True, exist_ok=True)

    rc = run_pip([
        "install", "--no-index", "--find-links", str(WHEELS_DIR),
        "--target", str(PKGS_DIR),
        *REQUIREMENTS_TORCH, *REQUIREMENTS_REST
    ])
    if rc != 0:
        print("[WARN] Offline constraints failed, trying all *.whl directly (excluding ORT for now)…")
        wheel_files = [p for p in sorted(WHEELS_DIR.glob("*.whl")) if not p.name.startswith("onnxruntime")]
        if not wheel_files:
            print("[ERR] No .whl files in ./wheels.")
            return False
        rc2 = run_pip(["install", "--no-index", "--target", str(PKGS_DIR)] + [str(p) for p in wheel_files])
        if rc2 != 0:
            print("[ERR] Offline install from wheel files failed.")
            return False

    # ONNX Runtime: prefer GPU, fallback CPU
    install_onnxruntime_prefer_gpu_offline()

    return have_all_imports(verbose=True)

def force_update_ultralytics(online: bool) -> bool:
    """Force reinstall ultralytics to a YOLOv11-capable version."""
    print("\n=== Force (re)install ultralytics for YOLOv11 ===")
    run_pip(["uninstall", "-y", "ultralytics"])

    if online:
        rc = run_pip(["install", "--target", str(PKGS_DIR), "ultralytics>=8.3.0"])
    else:
        if not WHEELS_DIR.exists():
            print("[ERR] wheels/ missing; cannot offline-install ultralytics.")
            return False
        whls = sorted([p for p in WHEELS_DIR.glob("ultralytics-*.whl")], reverse=True)
        if not whls:
            print("[ERR] No ultralytics wheel in ./wheels.")
            return False
        rc = run_pip(["install", "--no-index", "--target", str(PKGS_DIR), str(whls[0])])

    if rc != 0:
        print("[ERR] ultralytics reinstall failed.")
        return False

    add_pkgs_to_syspath()
    ok = ultralytics_supports_yolo11()
    print(f"[CHK] YOLOv11 support (C3k2): {'OK' if ok else 'NO'}")
    return ok

def print_ort_provider_info():
    try:
        add_pkgs_to_syspath()
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"[ORT] Available providers: {providers}")
        if "CUDAExecutionProvider" in providers:
            print("[ORT] Using GPU is possible (CUDAExecutionProvider present).")
        else:
            print("[ORT] GPU provider not available; ONNX will run on CPU.")
    except Exception as e:
        print(f"[ORT] Could not import onnxruntime to query providers: {e}")

def main():
    print("=== bootstrap_env.py ===")
    print(f"Python: {sys.executable}")
    print(f"Project: {BASE}")
    print(f"PKGS_DIR: {PKGS_DIR}")
    print(f"WHEELS_DIR: {WHEELS_DIR}")
    print(f"Target app: {APP_PATH.name}\n")

    add_pkgs_to_syspath()

    # 0) Quick check
    have = have_all_imports(verbose=True)
    online = is_online()

    if not have:
        if online:
            print("\n[INFO] Network available -> trying online install…")
            ok = install_online()
            if not ok:
                print("\n[INFO] Online incomplete -> trying offline wheels…")
                ok = install_offline_from_wheels()
        else:
            print("\n[INFO] No network -> trying offline wheels…")
            ok = install_offline_from_wheels()

        if not ok:
            print("\n[FATAL] Could not install all required packages.")
            sys.exit(1)

    # ONNX Runtime provider info
    print_ort_provider_info()

    # 1) YOLOv11 sanity (C3k2 present?)
    if not ultralytics_supports_yolo11():
        print("[INFO] Current ultralytics seems too old for YOLOv11 (missing C3k2).")
        if not force_update_ultralytics(online=online):
            print("\n[FATAL] ultralytics still misses YOLOv11 support.")
            print(" - If offline, place a recent ultralytics wheel in ./wheels and rerun.")
            sys.exit(1)

    # 2) Launch app
    if not APP_PATH.exists():
        print(f"\n[ERR] {APP_PATH.name} not found next to bootstrap_env.py")
        sys.exit(1)

    print("\n=== Launching", APP_PATH.name, "===\n")
    runpy.run_path(str(APP_PATH), run_name="__main__")

if __name__ == "__main__":
    main()
