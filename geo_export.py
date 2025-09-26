# geo_export.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any

# Optional tiny dependency to read GeoTIFF tags (install: tifffile)
try:
    import tifffile
except Exception:
    tifffile = None

# ---------- Affine helpers ----------
# Worldfile order: A, D, B, E, C, F  (see ESRI spec)
# Mapping: X = A*col + B*row + C ;  Y = D*col + E*row + F
Affine = Tuple[float, float, float, float, float, float]

_WF_EXTS = {
    ".wld", ".jgw", ".jpgw", ".jpegw", ".pgw", ".pngw",
    ".tfw", ".gfw", ".gifw", ".bpw", ".tifw"
}

def _find_worldfile(img_path: Path) -> Optional[Path]:
    base = img_path.with_suffix("")  # remove .ext
    # try known specific first, then generic .wld
    candidates = []
    ext = img_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}: candidates += [base.with_suffix(".jgw"), base.with_suffix(".jpgw"), base.with_suffix(".jpegw")]
    if ext == ".png":             candidates += [base.with_suffix(".pgw"), base.with_suffix(".pngw")]
    if ext in {".tif", ".tiff"}:  candidates += [base.with_suffix(".tfw"), base.with_suffix(".tifw")]
    candidates += [base.with_suffix(".gfw"), base.with_suffix(".bpw"), base.with_suffix(".wld")]
    for c in candidates:
        if c.exists(): return c
    # last pass: any sibling with worldfile-ish extension
    for s in img_path.parent.iterdir():
        if s.suffix.lower() in _WF_EXTS and s.stem == base.name:
            return s
    return None

def _load_worldfile(fp: Path) -> Affine:
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
            mt = tags.get("ModelTransformationTag")  # 34264
            if mt:
                m = list(mt.value)  # 16 numbers
                # 4x4 row-major: [m00 m01 m02 m03 m10 m11 ...]
                A = m[0]; B = m[1]; C = m[3]
                D = m[4]; E = m[5]; F = m[7]
                return (A, B, C, D, E, F)
            ps = tags.get("ModelPixelScaleTag")  # 33550
            tp = tags.get("ModelTiepointTag")   # 33922
            if ps and tp:
                sx, sy = float(ps.value[0]), float(ps.value[1])
                # Use first tiepoint
                vals = list(tp.value)
                i, j, _k, X0, Y0, _Z0 = vals[0:6]
                # north-up assumption: Y decreases with row
                A, B, C = sx, 0.0, X0 - i * sx
                D, E, F = 0.0, -sy, Y0 + j * sy
                return (A, B, C, D, E, F)
    except Exception:
        return None
    return None

def get_affine(img_path: Path) -> Optional[Affine]:
    """Return affine transform (A,B,C,D,E,F) or None if not georeferenced."""
    img_path = Path(img_path)
    # 1) worldfile
    wf = _find_worldfile(img_path)
    if wf:
        try:
            return _load_worldfile(wf)
        except Exception:
            pass
    # 2) GeoTIFF tags
    if img_path.suffix.lower() in {".tif", ".tiff"}:
        af = _affine_from_geotiff(img_path)
        if af: return af
    return None

def pix2geo(aff: Affine, col: float, row: float) -> Tuple[float, float]:
    A,B,C,D,E,F = aff
    X = A*col + B*row + C
    Y = D*col + E*row + F
    return (X, Y)

# ---------- GeoJSON writing ----------
def _feat_point(x: float, y: float, props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type":"Feature","geometry":{"type":"Point","coordinates":[x,y]},"properties":props}

def _feat_poly(coords: Iterable[Tuple[float,float]], props: Dict[str, Any]) -> Dict[str, Any]:
    ring = [[float(x), float(y)] for (x,y) in coords]
    if ring[0] != ring[-1]: ring.append(ring[0])
    return {"type":"Feature","geometry":{"type":"Polygon","coordinates":[ring]},"properties":props}

def _bbox_to_poly(px: float, py: float, qx: float, qy: float, aff: Affine):
    # (x1,y1,x2,y2) in pixel -> polygon in map
    x1,y1 = pix2geo(aff, px, py)
    x2,y2 = pix2geo(aff, qx, py)
    x3,y3 = pix2geo(aff, qx, qy)
    x4,y4 = pix2geo(aff, px, qy)
    return [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]

def export_geojson_for_image(
    image_path: Path,
    detections: Iterable[Dict[str, Any]],
    aois: Optional[Iterable[Dict[str, Any]]],
    out_dir: Path,
    crs_hint: Optional[str] = None,
) -> Optional[Path]:
    """
    detections: iterable of dicts with at least:
        {'cls': 'car', 'conf': 0.87, 'bbox': [x1,y1,x2,y2]} ; 'centroid': [cx,cy] optional
    aois: iterable of dicts with:
        {'name': 'Zone A', 'polygon': [(x,y),...]}  # pixel coords
    Writes <image>__detections.geojson and <image>__aois.geojson (if aois provided).
    Returns detections geojson path or None if not georeferenced.
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    aff = get_affine(image_path)
    if not aff:
        return None

    # detections
    feats = []
    for det in detections:
        cls = det.get("cls") or det.get("class") or det.get("label")
        conf = float(det.get("conf", det.get("confidence", 0.0)))
        bbox = det.get("bbox") or det.get("box")
        if bbox is None or len(bbox) != 4:
            continue
        x1,y1,x2,y2 = [float(v) for v in bbox]
        # centroid
        cx, cy = det.get("centroid", ((x1+x2)/2.0, (y1+y2)/2.0))
        gx, gy = pix2geo(aff, cx, cy)
        feats.append(_feat_point(gx, gy, {"type":"centroid","cls":cls,"conf":conf,"image":image_path.name}))
        # bbox polygon
        poly = _bbox_to_poly(x1,y1,x2,y2, aff)
        feats.append(_feat_poly(poly, {"type":"bbox","cls":cls,"conf":conf,"image":image_path.name}))

    # aois
    if aois:
        aoi_feats = []
        for a in aois:
            name = a.get("name","AOI")
            poly_px = a.get("polygon") or a.get("points") or []
            poly_geo = [pix2geo(aff, float(px), float(py)) for (px,py) in poly_px]
            aoi_feats.append(_feat_poly(poly_geo, {"type":"aoi","name":name,"image":image_path.name}))
        aoi_path = out_dir / f"{image_path.stem}__aois.geojson"
        aoi_payload = {"type":"FeatureCollection","features":aoi_feats}
        if crs_hint: aoi_payload["properties"]={"crs": crs_hint}
        aoi_path.write_text(json.dumps(aoi_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # detections path
    det_path = out_dir / f"{image_path.stem}__detections.geojson"
    payload = {"type":"FeatureCollection","features":feats}
    if crs_hint:
        payload["properties"]={"crs": crs_hint}
    det_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return det_path
