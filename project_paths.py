# project_paths.py
"""Central project paths and local package path helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_ROOT / "input"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
PRESETS_DIR = PROJECT_ROOT / "presets"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
TOOLS_DIR = PROJECT_ROOT / "tools"

PKGS_DIR = PROJECT_ROOT / "_pkgs"
LEGACY_PKGS_DIR = PROJECT_ROOT / "pkgs"
LOCAL_PACKAGE_DIRS = (PKGS_DIR, LEGACY_PKGS_DIR)

WORKING_DIRS = (INPUT_DIR, MODELS_DIR, OUTPUT_DIR, PRESETS_DIR)


def _same_path(a: str | Path, b: Path) -> bool:
    try:
        return Path(a).resolve() == b.resolve()
    except OSError:
        return False


def _is_readable_package_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    with os.scandir(entry.path):
                        pass
                else:
                    with open(entry.path, "rb"):
                        pass
        return True
    except OSError:
        return False


def add_local_package_paths(strict: bool = False) -> list[Path]:
    """
    Prepend local dependency folders to sys.path.

    Unreadable local package folders are skipped, which avoids hard failures
    from partially synced or permission-blocked OneDrive folders.

    strict=True also strips non-local site-packages entries, but only when at
    least one readable local package folder exists.
    """
    added: list[Path] = []
    existing_dirs = [p for p in LOCAL_PACKAGE_DIRS if _is_readable_package_dir(p)]

    for p in reversed(existing_dirs):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
        added.append(p)

    if strict and existing_dirs:
        keep: list[str] = []
        for sp in sys.path:
            low = sp.replace("\\", "/").lower()
            if "site-packages" in low or "dist-packages" in low:
                if any(_same_path(sp, local_dir) for local_dir in existing_dirs):
                    keep.append(sp)
            else:
                keep.append(sp)
        sys.path[:] = keep

    return list(reversed(added))


def ensure_working_dirs() -> None:
    """Create local app working folders without touching their contents."""
    for p in WORKING_DIRS:
        p.mkdir(parents=True, exist_ok=True)
