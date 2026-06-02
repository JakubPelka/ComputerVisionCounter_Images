# aoi_utils.py
"""AOI normalization helpers shared by UI and inference code."""

from __future__ import annotations

from typing import Any

Point = list[float]
NormalizedAOI = dict[str, list[Point] | str]


def _coerce_polygon(points: Any) -> list[Point]:
    if not points:
        return []
    out: list[Point] = []
    try:
        for pt in points:
            x, y = pt
            out.append([float(x), float(y)])
    except (TypeError, ValueError):
        return []
    return out if len(out) >= 3 else []


def _poly_from_mapping(data: dict[str, Any]) -> Any:
    return data.get("polygon") or data.get("points") or data.get("pts")


def normalize_aois(
    data: Any,
    *,
    item_default_name: str = "AOI",
    single_default_name: str = "AOI 1",
    point_list_name: str = "AOI 1",
    indexed_item_names: bool = False,
) -> list[dict[str, Any]]:
    """
    Normalize legacy AOI shapes to [{'name': str, 'polygon': [[x, y], ...]}].

    Supported inputs include:
    - {'aois': [{'name': ..., 'polygon'/'points'/'pts': ...}]}
    - {'name': ..., 'polygon'/'points'/'pts': ...}
    - {'Zone A': [[x, y], ...], 'Zone B': {'points': ...}}
    - [{'name': ..., 'polygon'/'points'/'pts': ...}]
    - [[x, y], [x, y], [x, y]]
    """
    if not data:
        return []

    out: list[dict[str, Any]] = []

    def item_name(i: int, raw_name: Any = None) -> str:
        if raw_name:
            return str(raw_name)
        if indexed_item_names:
            return f"AOI {i}"
        return item_default_name

    def add(name: Any, points: Any) -> None:
        poly = _coerce_polygon(points)
        if poly:
            out.append({"name": str(name), "polygon": poly})

    if isinstance(data, dict):
        if isinstance(data.get("aois"), list):
            for i, item in enumerate(data["aois"], 1):
                if isinstance(item, dict):
                    add(item_name(i, item.get("name")), _poly_from_mapping(item))
                else:
                    add(item_name(i), item)
            return out

        direct_poly = _poly_from_mapping(data)
        if direct_poly:
            add(data.get("name", single_default_name), direct_poly)
            return out

        for key, value in data.items():
            if isinstance(value, dict):
                add(key, _poly_from_mapping(value))
            else:
                add(key, value)
        return out

    if isinstance(data, list):
        if data and isinstance(data[0], (list, tuple)) and len(data[0]) == 2:
            add(point_list_name, data)
            return out
        for i, item in enumerate(data, 1):
            if isinstance(item, dict):
                add(item_name(i, item.get("name")), _poly_from_mapping(item))
            else:
                add(item_name(i), item)

    return out


def normalize_aois_as_tuples(
    data: Any,
    **kwargs: Any,
) -> list[tuple[str, list[Point]]]:
    return [(str(a["name"]), a["polygon"]) for a in normalize_aois(data, **kwargs)]
