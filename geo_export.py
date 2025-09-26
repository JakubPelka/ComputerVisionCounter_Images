# geo_export.py — CSV-only GIS export (points & boxes as WKT; AOIs as WKT)
from __future__ import annotations
import json, math, re
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any

try:
    import tifffile  # optional, for GeoTIFF georeferencing
except Exception:
    tifffile = None

Affine = Tuple[float, float, float, float, float, float]
_WF_EXTS = {".wld", ".jgw", ".jpgw", ".jpegw", ".pgw", ".pngw", ".tfw", ".gfw", ".gifw", ".bpw", ".tifw"}

# -------- worldfile / geotiff helpers --------

def _find_worldfile(img_path: Path) -> Optional[Path]:
    base = img_path.with_suffix("")
    ext = img_path.suffix.lower()
    cand = []
    if ext in {".jpg",".jpeg"}: cand += [base.with_suffix(".jgw"), base.with_suffix(".jpgw"), base.with_suffix(".jpegw")]
    if ext == ".png": cand += [base.with_suffix(".pgw"), base.with_suffix(".pngw")]
    if ext in {".tif",".tiff"}: cand += [base.with_suffix(".tfw"), base.with_suffix(".tifw")]
    cand += [base.with_suffix(".gfw"), base.with_suffix(".bpw"), base.with_suffix(".wld")]
    for c in cand:
        if c.exists(): return c
    for s in img_path.parent.iterdir():
        if s.suffix.lower() in _WF_EXTS and s.stem == base.name:
            return s
    return None

def _load_worldfile(fp: Path) -> Affine:
    # ESRI order: A, D, B, E, C, F  -> convert to (A,B,C,D,E,F)
    vals = [float(x) for x in fp.read_text(encoding="utf-8").replace(",", ".").split()]
    if len(vals) != 6:
        raise ValueError(f"Worldfile must have 6 numbers, got {len(vals)} in {fp}")
    A, D, B, E, C, F = vals
    return (A, B, C, D, E, F)

def _affine_from_geotiff(fp: Path) -> Optional[Affine]:
    if not tifffile: return None
    try:
        with tifffile.TiffFile(str(fp)) as tf:
            tags = tf.pages[0].tags
            mt = tags.get("ModelTransformationTag")
            if mt:
                m = list(mt.value)
                A = m[0]; B = m[1]; C = m[3]
                D = m[4]; E = m[5]; F = m[7]
                return (A, B, C, D, E, F)
            ps = tags.get("ModelPixelScaleTag")
            tp = tags.get("ModelTiepointTag")
            if ps and tp:
                sx, sy = float(ps.value[0]), float(ps.value[1])
                vals = list(tp.value)
                i, j, _k, X0, Y0, _Z0 = vals[0:6]
                A, B, C = sx, 0.0, X0 - i * sx
                D, E, F = 0.0, -sy, Y0 + j * sy
                return (A, B, C, D, E, F)
    except Exception:
        return None
    return None

def get_affine(img_path: Path) -> Optional[Affine]:
    wf = _find_worldfile(img_path)
    if wf:
        try: return _load_worldfile(wf)
        except Exception: pass
    if img_path.suffix.lower() in {".tif",".tiff"}:
        af = _affine_from_geotiff(img_path)
        if af: return af
    return None

# -------- small utils --------

def pix2geo(aff: Affine, col: float, row: float) -> Tuple[float, float]:
    A,B,C,D,E,F = aff
    X = A*col + B*row + C
    Y = D*col + E*row + F
    return (X, Y)

def _finite_xy(x, y) -> bool:
    return (x is not None) and (y is not None) and math.isfinite(x) and math.isfinite(y)

def _csv_escape(s: str) -> str:
    s = "" if s is None else str(s)
    if any(ch in s for ch in [",", '"', "\n"]):
        return '"' + s.replace('"', '""') + '"'
    return s

def _poly_wkt(coords) -> str:
    # coords: [(x,y), ...] (no need to repeat start; we’ll close it)
    if not coords: return "POLYGON(())"
    ring = coords + ([coords[0]] if coords[0] != coords[-1] else [])
    coord_str = ", ".join(f"{x:.6f} {y:.6f}" for (x,y) in ring)
    return f"POLYGON(({coord_str}))"

# -------- public API (keeps original name; now CSV-only) --------

def export_geojson_for_image(  # noqa: API kept for start_app; now writes CSVs only
    image_path: Path,
    detections: Iterable[Dict[str, Any]],
    aois: Optional[Iterable[Dict[str, Any]]],
    out_dir: Path,
    crs_hint: Optional[str] = None,  # unused in CSV variant
) -> Optional[Path]:
    """
    CSV-only export:
      - <stem>__detections_p.csv  (image, cls, conf, x, y)
      - <stem>__detections_b.csv  (image, cls, conf, wkt)
      - <stem>__aois.csv          (image, name, wkt)
    Returns the path to the *points* CSV if written, else the boxes CSV, else None.
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    aff = get_affine(image_path)
    if not aff:
        # no georef -> nothing to write
        return None

    points_rows = []  # dicts with image,cls,conf,x,y
    boxes_rows  = []  # dicts with image,cls,conf,wkt

    # ---- detections -> world coords ----
    for det in (detections or []):
        cls  = det.get("cls") or det.get("class") or det.get("label")
        conf = float(det.get("conf", det.get("confidence", 0.0)))
        bbox = det.get("bbox") or det.get("box")
        if not bbox or len(bbox) != 4:
            continue
        x1,y1,x2,y2 = [float(v) for v in bbox]
        cx, cy = det.get("centroid", ((x1+x2)/2.0, (y1+y2)/2.0))

        # centroid
        gx, gy = pix2geo(aff, float(cx), float(cy))
        if _finite_xy(gx, gy):
            points_rows.append({"image": image_path.name, "cls": cls, "conf": conf, "x": gx, "y": gy})

        # bbox polygon -> WKT
        p1 = pix2geo(aff, x1, y1)
        p2 = pix2geo(aff, x2, y1)
        p3 = pix2geo(aff, x2, y2)
        p4 = pix2geo(aff, x1, y2)
        poly = [p1, p2, p3, p4]
        if all(_finite_xy(px,py) for (px,py) in poly):
            boxes_rows.append({"image": image_path.name, "cls": cls, "conf": conf, "wkt": _poly_wkt(poly)})

    # ---- AOIs -> WKT CSV ----
    if aois:
        aoi_rows = []
        for a in aois:
            name = a.get("name","AOI")
            poly_px = a.get("polygon") or a.get("points") or []
            poly_geo = [pix2geo(aff, float(px), float(py)) for (px,py) in poly_px]
            if all(_finite_xy(px,py) for (px,py) in poly_geo):
                aoi_rows.append({"image": image_path.name, "name": name, "wkt": _poly_wkt(poly_geo)})
        if aoi_rows:
            p = out_dir / f"{image_path.stem}__aois.csv"
            with p.open("w", encoding="utf-8") as f:
                f.write("image,name,wkt\n")
                for r in aoi_rows:
                    f.write(",".join([_csv_escape(r["image"]), _csv_escape(r["name"]), _csv_escape(r["wkt"])]) + "\n")

    points_path = None
    boxes_path = None

    # ---- write detections CSVs ----
    if points_rows:
        points_path = out_dir / f"{image_path.stem}__detections_p.csv"
        with points_path.open("w", encoding="utf-8") as f:
            f.write("image,cls,conf,x,y\n")
            for r in points_rows:
                f.write(",".join([
                    _csv_escape(r["image"]),
                    _csv_escape("" if r["cls"] is None else str(r["cls"])),
                    f"{float(r['conf']):.6f}",
                    f"{float(r['x']):.6f}",
                    f"{float(r['y']):.6f}",
                ]) + "\n")

    if boxes_rows:
        boxes_path = out_dir / f"{image_path.stem}__detections_b.csv"
        with boxes_path.open("w", encoding="utf-8") as f:
            f.write("image,cls,conf,wkt\n")
            for r in boxes_rows:
                f.write(",".join([
                    _csv_escape(r["image"]),
                    _csv_escape("" if r["cls"] is None else str(r["cls"])),
                    f"{float(r['conf']):.6f}",
                    _csv_escape(r["wkt"]),
                ]) + "\n")

    return points_path or boxes_path
