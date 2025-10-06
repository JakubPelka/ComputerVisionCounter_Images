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

# -------- utils --------

def _unique_path(p: Path) -> Path:
    """
    ## Non-destructive path: p, p_1, p_2, ...
    """
    p = Path(p)
    if not p.exists():
        return p
    d, stem, suf = p.parent, p.stem, p.suffix
    k = 1
    while True:
        q = d / f"{stem}_{k}{suf}"
        if not q.exists():
            return q
        k += 1

def _csv_escape(s: str) -> str:
    s = "" if s is None else str(s)
    if any(ch in s for ch in [",", '"', "\n", "\r"]):
        s = '"' + s.replace('"', '""') + '"'
    return s

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
        if s.suffix.lower() in _WF_EXTS and s.stem == img_path.stem:
            return s
    return None

def _read_worldfile(fp: Path) -> Optional[Affine]:
    try:
        vals = []
        with open(fp, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                vals.append(float(re.sub(r"[^\d\.\-eE+]", "", ln)))
        if len(vals) != 6:
            return None
        A, D, B, E, C, F = vals
        return (A, B, C, D, E, F)
    except Exception:
        return None

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
            # fallback: ModelPixelScale + ModelTiepoint
            mps = tags.get("ModelPixelScaleTag")
            mtp = tags.get("ModelTiepointTag")
            if mps and mtp:
                sx, sy, _ = list(mps.value)
                tie = list(mtp.value)
                # tiepoint format: (i,j,k, x,y,z) repeating; we use the first
                i, j, _k, X, Y, _Z = tie[:6]
                A = sx; B = 0.0; C = X - sx * i
                D = 0.0; E = -sy; F = Y + sy * j
                return (A, B, C, D, E, F)
    except Exception:
        return None
    return None

def get_affine(image_path: Path) -> Optional[Affine]:
    wf = _find_worldfile(image_path)
    if wf:
        return _read_worldfile(wf)
    if image_path.suffix.lower() in {".tif", ".tiff"}:
        return _affine_from_geotiff(image_path)
    return None

def pix2geo(aff: Affine, x: float, y: float) -> Tuple[float, float]:
    A, B, C, D, E, F = aff
    gx = A * x + B * y + C
    gy = D * x + E * y + F
    return gx, gy

def _finite_xy(x: float, y: float) -> bool:
    return math.isfinite(float(x)) and math.isfinite(float(y))

def _poly_wkt(poly_xy: Iterable[Tuple[float,float]]) -> str:
    pts = ", ".join(f"{x:.6f} {y:.6f}" for (x,y) in poly_xy)
    return f"POLYGON(({pts}))"

# -------- main export --------

def export_geojson_for_image(  # noqa: API kept for start_app; now writes CSVs only
    image_path: Path,
    detections: Iterable[Dict[str, Any]],
    aois: Optional[Iterable[Dict[str, Any]]],
    out_dir: Path,
    crs_hint: Optional[str] = None,  # unused in CSV variant
) -> Optional[Path]:
    """
    CSV-only export:
      - <stem>__detections_point.csv  (image, cls, conf, x, y, aoi)
      - <stem>__detections_box.csv    (image, cls, conf, aoi, wkt)
      - <stem>__aois.csv              (image, name, wkt)
    Returns the path to the *point* CSV if written, else the box CSV, else None.
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    aff = get_affine(image_path)
    if not aff:
        # no georef -> nothing to write
        return None

    # row dicts:
    #  * points_rows: {"image","cls","conf","x","y","aoi"}
    #  * boxes_rows:  {"image","cls","conf","aoi","wkt"}
    points_rows = []
    boxes_rows  = []

    # ---- detections -> world coords ----
    for det in (detections or []):
        cls  = det.get("cls") or det.get("class") or det.get("label")
        conf = float(det.get("conf", det.get("confidence", 0.0)))
        bbox = det.get("bbox") or det.get("box")
        if not bbox or len(bbox) != 4:
            continue
        x1,y1,x2,y2 = [float(v) for v in bbox]
        cx, cy = det.get("centroid", ((x1+x2)/2.0, (y1+y2)/2.0))

        # AOI name propagated from runner/normalizer. Allow both keys.
        aoi_name = (det.get("aoi") or det.get("aoi_name") or "")

        # centroid -> point CSV
        gx, gy = pix2geo(aff, float(cx), float(cy))
        if _finite_xy(gx, gy):
            points_rows.append({
                "image": image_path.name,
                "cls": cls,
                "conf": conf,
                "x": gx,
                "y": gy,
                "aoi": aoi_name,
            })

        # bbox polygon -> WKT (box CSV)
        p1 = pix2geo(aff, x1, y1)
        p2 = pix2geo(aff, x2, y1)
        p3 = pix2geo(aff, x2, y2)
        p4 = pix2geo(aff, x1, y2)
        poly = [p1, p2, p3, p4]
        if all(_finite_xy(px,py) for (px,py) in poly):
            boxes_rows.append({
                "image": image_path.name,
                "cls": cls,
                "conf": conf,
                "aoi": aoi_name,
                "wkt": _poly_wkt(poly),
            })

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
            p = _unique_path(out_dir / f"{image_path.stem}__aois.csv")
            with p.open("w", encoding="utf-8") as f:
                f.write("image,name,wkt\n")
                for r in aoi_rows:
                    f.write(",".join([_csv_escape(r["image"]), _csv_escape(r["name"]), _csv_escape(r["wkt"])]) + "\n")

    point_path = None
    box_path = None

    # ---- write detections CSVs (unique names) ----
    if points_rows:
        point_path = _unique_path(out_dir / f"{image_path.stem}__detections_point.csv")
        with point_path.open("w", encoding="utf-8") as f:
            f.write("image,cls,conf,x,y,aoi\n")
            for r in points_rows:
                f.write(",".join([
                    _csv_escape(r["image"]),
                    _csv_escape("" if r["cls"] is None else str(r["cls"])),
                    f"{float(r['conf']):.6f}",
                    f"{float(r['x']):.6f}",
                    f"{float(r['y']):.6f}",
                    _csv_escape(r.get("aoi","")),
                ]) + "\n")

    if boxes_rows:
        box_path = _unique_path(out_dir / f"{image_path.stem}__detections_box.csv")
        with box_path.open("w", encoding="utf-8") as f:
            f.write("image,cls,conf,aoi,wkt\n")
            for r in boxes_rows:
                f.write(",".join([
                    _csv_escape(r["image"]),
                    _csv_escape("" if r["cls"] is None else str(r["cls"])),
                    f"{float(r['conf']):.6f}",
                    _csv_escape(r.get("aoi","")),
                    _csv_escape(r["wkt"]),
                ]) + "\n")

    return point_path or box_path
