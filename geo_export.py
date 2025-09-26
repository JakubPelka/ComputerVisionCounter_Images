# geo_export.py — split detections into points (_p) and boxes (_b), no top-level CRS
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any

try:
    import tifffile
except Exception:
    tifffile = None

Affine = Tuple[float, float, float, float, float, float]
_WF_EXTS = {".wld", ".jgw", ".jpgw", ".jpegw", ".pgw", ".pngw", ".tfw", ".gfw", ".gifw", ".bpw", ".tifw"}

# ---------- Worldfile / GeoTIFF helpers ----------

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
    # fallback: any matching name + known worldfile extension
    for s in img_path.parent.iterdir():
        if s.suffix.lower() in _WF_EXTS and s.stem == base.name:
            return s
    return None

def _load_worldfile(fp: Path) -> Affine:
    # ESRI worldfile order: A, D, B, E, C, F
    vals = [float(x) for x in fp.read_text(encoding="utf-8").replace(",", ".").split()]
    if len(vals) != 6:
        raise ValueError(f"Worldfile must have 6 numbers, got {len(vals)} in {fp}")
    A, D, B, E, C, F = vals
    # We use (A,B,C,D,E,F)
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
            # Fallback: PixelScale + Tiepoint (GDAL-style)
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
        try: 
            return _load_worldfile(wf)
        except Exception:
            pass
    if img_path.suffix.lower() in {".tif",".tiff"}:
        af = _affine_from_geotiff(img_path)
        if af: return af
    return None

def pix2geo(aff: Affine, col: float, row: float) -> Tuple[float, float]:
    A,B,C,D,E,F = aff
    X = A*col + B*row + C
    Y = D*col + E*row + F
    return (X, Y)

# ---------- GeoJSON helpers ----------

def _finite_xy(x, y) -> bool:
    return (x is not None) and (y is not None) and math.isfinite(x) and math.isfinite(y)

def _feat_point(x: float, y: float, props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type":"Feature","geometry":{"type":"Point","coordinates":[x,y]},"properties":props}

def _feat_poly(coords, props: Dict[str, Any]) -> Dict[str, Any]:
    ring = [[float(x), float(y)] for (x,y) in coords]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type":"Feature","geometry":{"type":"Polygon","coordinates":[ring]},"properties":props}

def _bbox_to_poly(px: float, py: float, qx: float, qy: float, aff: Affine):
    x1,y1 = pix2geo(aff, px, py)
    x2,y2 = pix2geo(aff, qx, py)
    x3,y3 = pix2geo(aff, qx, qy)
    x4,y4 = pix2geo(aff, px, qy)
    return [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]

# ---------- Public API ----------

def export_geojson_for_image(
    image_path: Path,
    detections: Iterable[Dict[str, Any]],
    aois: Optional[Iterable[Dict[str, Any]]],
    out_dir: Path,
    crs_hint: Optional[str] = None,  # accepted but not written (we omit CRS to match your “works best” case)
) -> Optional[Path]:
    """
    Writes:
      - <stem>__detections_p.geojson  (centroids as Points)
      - <stem>__detections_b.geojson  (boxes as Polygons)
      - <stem>__aois.geojson          (AOI polygons)
    Returns path to the *points* file if written, else boxes, else None.
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    aff = get_affine(image_path)
    if not aff:
        # No georeferencing found -> skip exporting GeoJSON
        return None

    feats_p: list[Dict[str, Any]] = []
    feats_b: list[Dict[str, Any]] = []

    for det in (detections or []):
        cls = det.get("cls") or det.get("class") or det.get("label")
        conf = float(det.get("conf", det.get("confidence", 0.0)))
        bbox = det.get("bbox") or det.get("box")
        if not bbox or len(bbox) != 4:
            continue
        x1,y1,x2,y2 = [float(v) for v in bbox]
        cx, cy = det.get("centroid", ((x1+x2)/2.0, (y1+y2)/2.0))

        # point
        gx, gy = pix2geo(aff, float(cx), float(cy))
        if _finite_xy(gx, gy):
            feats_p.append(_feat_point(gx, gy, {
                "type":"centroid","cls":cls,"conf":conf,"image":image_path.name
            }))

        # polygon
        poly = _bbox_to_poly(x1,y1,x2,y2, aff)
        if all(_finite_xy(px,py) for (px,py) in poly):
            feats_b.append(_feat_poly(poly, {
                "type":"bbox","cls":cls,"conf":conf,"image":image_path.name
            }))

    # AOIs (separate file, optional)
    if aois:
        aoi_feats = []
        for a in aois:
            name = a.get("name","AOI")
            poly_px = a.get("polygon") or a.get("points") or []
            poly_geo = [pix2geo(aff, float(px), float(py)) for (px,py) in poly_px]
            if all(_finite_xy(px,py) for (px,py) in poly_geo):
                aoi_feats.append(_feat_poly(poly_geo, {"type":"aoi","name":name,"image":image_path.name}))
        if aoi_feats:
            (out_dir / f"{image_path.stem}__aois.geojson").write_text(
                json.dumps({"type":"FeatureCollection","features":aoi_feats}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

    points_path = None
    boxes_path = None

    if feats_p:
        points_path = out_dir / f"{image_path.stem}__detections_p.geojson"
        points_path.write_text(
            json.dumps({"type":"FeatureCollection","features":feats_p}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    if feats_b:
        boxes_path = out_dir / f"{image_path.stem}__detections_b.geojson"
        boxes_path.write_text(
            json.dumps({"type":"FeatureCollection","features":feats_b}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    return points_path or boxes_path
