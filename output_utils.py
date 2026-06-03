# output_utils.py
"""Shared helpers for run output files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

DETECTIONS_FULL_CSV = "detections_full.csv"
RESULTS_PER_IMAGE_CSV = "results_per_image.csv"
RESULTS_TOTALS_JSON = "results_totals.json"
RUN_METADATA_JSON = "run_metadata.json"

GIS_AOIS_CSV_SUFFIX = "__aois.csv"
GIS_DETECTIONS_POINT_CSV_SUFFIX = "__detections_point.csv"
GIS_DETECTIONS_BOX_CSV_SUFFIX = "__detections_box.csv"


def suffixed_csv_name(stem: str, suffix: str) -> str:
    return f"{stem}{suffix}"


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def unique_path(path: Path | str) -> Path:
    """Avoid overwriting by appending _1, _2, ..."""
    p = Path(path)
    if not p.exists():
        return p
    parent, stem, suffix = p.parent, p.stem, p.suffix
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def write_csv(
    rows: Sequence[Sequence[Any]],
    out_path: Path | str,
    header: Optional[Sequence[Any]] = None,
    *,
    unique: bool = True,
) -> Path:
    path = Path(out_path)
    ensure_dir(path.parent)
    if unique:
        path = unique_path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)
    return path


def write_json(obj: Any, out_path: Path | str, *, unique: bool = True) -> Path:
    path = Path(out_path)
    ensure_dir(path.parent)
    if unique:
        path = unique_path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def class_names_from_counts(per_image_counts: Sequence[tuple[Any, dict[str, int]]]) -> list[str]:
    return sorted({name for _, counts in per_image_counts for name in counts.keys()})


def per_image_count_table(
    per_image_counts: Sequence[tuple[Any, dict[str, int]]],
    *,
    image_header: str = "image",
) -> tuple[list[str], list[list[Any]]]:
    class_names = class_names_from_counts(per_image_counts)
    header = [image_header] + class_names
    rows = [[image_name] + [counts.get(name, 0) for name in class_names] for image_name, counts in per_image_counts]
    return header, rows
