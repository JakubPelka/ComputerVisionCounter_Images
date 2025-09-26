# geo_export.py — export detections to GeoJSON (points/boxes), CSV, plain JSON, and optional KML (WGS84)
from __future__ import annotations
import json, math, re
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any

try:
    import tifffile  # for GeoTIFF georeferencing
except Exception:
    tifffile = None

try:
    # only used when writing KML (reproject to EPSG:4326)
    from pyproj import CRS, Transformer  # type: ignore
except Exception:
    CRS = None
    Transformer = None

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
    for s in img_path.parent.iterdir():
        if s.suffix.lower() in _WF_EXTS and s.stem == base.name:
            return s
    return None

def _load_worldfile(fp: Path) -> Affine:
    vals = [float(x) for x in fp.read_text(encoding="utf-8").replace(",", ".").split()]
    if len(vals) != 6:
        raise ValueError(f"Worldfile must have 6 numbers, got {len(vals)} in {fp}")
    # ESRI order: A, D, B, E, C, F  -> convert to (A,B,C,D,E,F)
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
            # GDAL-style: PixelScale + Tiepoint
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

# ---------- CRS helpers ----------

def _epsg_from_prj(prj_path: Path) -> Optional[str]:
    try:
        txt = prj_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try: txt = prj_path.read_text(encoding="latin-1", errors="ignore")
        except Exception: return None
    m = re.search(r'AUTHORITY\["EPSG","(\d+)"\]', txt)
    return m.group(1) if m else None

def _make_wgs84_transformer(epsg: Optional[str]):
    if not (epsg and CRS and Transformer):
        return None
    try:
        crs_src = CRS.from_epsg(int(epsg))
        crs_dst = CRS.from_epsg(4326)
        return Transformer.from_crs(crs_src, crs_dst, always_xy=True)
    except Exception:
        return None

# ---------- GeoJSON helpers ----------

def pix2geo(aff: Affine, col: float, row: float) -> Tuple[float, float]:
    A,B,C,D,E,F = aff
    X = A*col + B*row + C
    Y = D*col + E*row + F
    return (X, Y)

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

# ---------- Simple writers ----------

def _write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

def _write_json(path: Path, payload: dict):
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))

def _csv_escape(s: str) -> str:
    if any(ch in s for ch in [",", '"', "\n"]):
        return '"' + s.replace('"', '""') + '"'
    return s

# ---------- Public API ----------

def export_geojson_for_image(
    image_path: Path,
    detections: Iterable[Dict[str, Any]],
    aois: Optional[Iterable[Dict[str, Any]]],
    out_dir: Path,
    crs_hint: Optional[str] = None,  # optional EPSG like "EPSG:3007" (used only for KML reprojection)
) -> Optional[Path]:
    """
    Writes (if data present):
      - <stem>__detections_p.geojson / .csv         (points)
      - <stem>__detections_b.geojson / .csv         (boxes; CSV has WKT)
      - <stem>__detections.json                      (plain JSON: points+boxes)
      - <stem>__detections_wgs84.kml                 (if reprojection available)
      - <stem>__aois.geojson                         (AOI polygons)
    Returns path to the *points* GeoJSON if written, else boxes GeoJSON, else None.
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    aff = get_affine(image_path)
    if not aff:
        return None

    # EPSG (for optional KML reprojection)
    epsg = None
    if crs_hint:
        m = re.search(r"EPSG[:]*[:]*\s*(\d+)", crs_hint, re.IGNORECASE)
        if m: epsg = m.group(1)
    if not epsg:
        prj = image_path.with_suffix(".prj")
        if prj.exists():
            epsg = _epsg_from_prj(prj)
    to_wgs84 = _make_wgs84_transformer(epsg)

    feats_p: list[Dict[str, Any]] = []
    feats_b: list[Dict[str, Any]] = []
    plain_points: list[dict] = []
    plain_boxes: list[dict] = []

    # Detections -> world coords
    for det in (detections or []):
        cls = det.get("cls") or det.get("class") or det.get("label")
        conf = float(det.get("conf", det.get("confidence", 0.0)))
        bbox = det.get("bbox") or det.get("box")
        if not bbox or len(bbox) != 4:
            continue
        x1,y1,x2,y2 = [float(v) for v in bbox]
        cx, cy = det.get("centroid", ((x1+x2)/2.0, (y1+y2)/2.0))

        # centroid
        gx, gy = pix2geo(aff, float(cx), float(cy))
        if _finite_xy(gx, gy):
            feats_p.append(_feat_point(gx, gy, {"type":"centroid","cls":cls,"conf":conf,"image":image_path.name}))
            plain_points.append({"x": gx, "y": gy, "cls": cls, "conf": conf, "image": image_path.name})

        # bbox polygon
        poly = _bbox_to_poly(x1,y1,x2,y2, aff)
        if all(_finite_xy(px,py) for (px,py) in poly):
            feats_b.append(_feat_poly(poly, {"type":"bbox","cls":cls,"conf":conf,"image":image_path.name}))
            plain_boxes.append({"poly": poly, "cls": cls, "conf": conf, "image": image_path.name})

    # ----- AOIs (GeoJSON; no CRS to avoid mismatches) -----
    if aois:
        aoi_feats = []
        for a in aois:
            name = a.get("name","AOI")
            poly_px = a.get("polygon") or a.get("points") or []
            poly_geo = [pix2geo(aff, float(px), float(py)) for (px,py) in poly_px]
            if all(_finite_xy(px,py) for (px,py) in poly_geo):
                aoi_feats.append(_feat_poly(poly_geo, {"type":"aoi","name":name,"image":image_path.name}))
        if aoi_feats:
            _write_json(out_dir / f"{image_path.stem}__aois.geojson",
                        {"type":"FeatureCollection","features":aoi_feats})

    # ----- GeoJSON (points / boxes) -----
    points_path = None
    boxes_path = None
    if feats_p:
        points_path = out_dir / f"{image_path.stem}__detections_p.geojson"
        _write_json(points_path, {"type":"FeatureCollection","features":feats_p})
    if feats_b:
        boxes_path = out_dir / f"{image_path.stem}__detections_b.geojson"
        _write_json(boxes_path, {"type":"FeatureCollection","features":feats_b})

    # ----- CSVs -----
    if plain_points:
        csvp = out_dir / f"{image_path.stem}__detections_p.csv"
        with csvp.open("w", encoding="utf-8") as f:
            f.write("image,cls,conf,x,y\n")
            for r in plain_points:
                f.write(",".join([
                    _csv_escape(r["image"]),
                    _csv_escape(str(r["cls"])),
                    str(r["conf"]),
                    f"{r['x']:.6f}",
                    f"{r['y']:.6f}",
                ]) + "\n")
    if plain_boxes:
        csvb = out_dir / f"{image_path.stem}__detections_b.csv"
        with csvb.open("w", encoding="utf-8") as f:
            f.write("image,cls,conf,wkt\n")
            for r in plain_boxes:
                poly = r["poly"]
                # WKT polygon (no CRS info here; QGIS can load by choosing WKT geometry)
                coords = ", ".join([f"{x:.6f} {y:.6f}" for (x,y) in (poly + [poly[0]])])
                wkt = f"POLYGON(({coords}))"
                f.write(",".join([
                    _csv_escape(r["image"]),
                    _csv_escape(str(r["cls"])),
                    str(r["conf"]),
                    _csv_escape(wkt),
                ]) + "\n")

    # ----- Plain JSON (easy to inspect) -----
    _write_json(out_dir / f"{image_path.stem}__detections.json",
                {"image": image_path.name, "points": plain_points, "boxes": plain_boxes})

    # ----- Optional KML (WGS84) -----
    if to_wgs84 and (plain_points or plain_boxes):
        kml = [ '<?xml version="1.0" encoding="UTF-8"?>',
                '<kml xmlns="http://www.opengis.net/kml/2.2">',
                "  <Document>",
                f"    <name>{image_path.stem} detections</name>" ]
        # points
        for r in plain_points:
            lon, lat = to_wgs84.transform(r["x"], r["y"])
            kml += [
                "    <Placemark>",
                f"      <name>{r['cls']} ({r['conf']:.2f})</name>",
                "      <Point>",
                f"        <coordinates>{lon:.8f},{lat:.8f},0</coordinates>",
                "      </Point>",
                "    </Placemark>"
            ]
        # boxes
        for r in plain_boxes:
            ring_ll = [to_wgs84.transform(x, y) for (x,y) in (r["poly"] + [r["poly"][0]])]
            coord_str = " ".join([f"{lon:.8f},{lat:.8f},0" for (lon,lat) in ring_ll])
            kml += [
                "    <Placemark>",
                f"      <name>{r['cls']} ({r['conf']:.2f})</name>",
                "      <Polygon>",
                "        <outerBoundaryIs>",
                "          <LinearRing>",
                f"            <coordinates>{coord_str}</coordinates>",
                "          </LinearRing>",
                "        </outerBoundaryIs>",
                "      </Polygon>",
                "    </Placemark>"
            ]
        kml += ["  </Document>", "</kml>"]
        _write_text(out_dir / f"{image_path.stem}__detections_wgs84.kml", "\n".join(kml))

    return points_path or boxes_path
