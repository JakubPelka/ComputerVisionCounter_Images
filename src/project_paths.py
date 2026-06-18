# project_paths.py
"""Central project paths and local package path helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "input"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
PRESETS_DIR = PROJECT_ROOT / "presets"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
TOOLS_DIR = PROJECT_ROOT / "tools"

WORKING_DIRS = (INPUT_DIR, MODELS_DIR, OUTPUT_DIR, PRESETS_DIR)





def ensure_working_dirs() -> None:
    """Create local app working folders without touching their contents."""
    for p in WORKING_DIRS:
        p.mkdir(parents=True, exist_ok=True)
