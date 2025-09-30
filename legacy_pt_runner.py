# legacy_pt_runner.py — YOLO .pt runner with robust AOI filtering and overlay
from __future__ import annotations

from pathlib import Path
from time import time
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

## Ultralytics must be available in your local env (_pkgs or pkgs)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # handled later

## Optional GIS exporter (graceful fallback if missing)
try:
    from geo_export import export_geojson_for_image
except Exception:
    export_geojson_for_image = None


# ------------------------------ small utils ------------------------------

def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def unique_path(p: Path) -> Path:
    """
    ## Non-destructive file naming
    Returns p if it doesn't exist; otherwise appends _1, _2, ...
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

def bbox_center(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))

def _iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    inter = w * h
    area_a = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

def _nms_numpy(xyxy: List[List[float]], scores: List[float], iou_thr: float = 0.55) -> List[int]:
    """
    ## Simple CPU NMS (keeps indices of selected boxes)
    """
    if not xyxy:
        return []
    idxs = np.argsort(np.asarray(scores))[::-1]
    arr = np.asarray(xyxy, dtype=np.float32)
    keep = []
    while len(idxs) > 0:
        i = int(idxs[0]); keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = np.array([_iou_xyxy(arr[i], arr[j]) for j in rest], dtype=np.float32)
        idxs = rest[ious <= iou_thr]
    return keep


# --------------------------- AOI helpers ---------------------------

def _normalize_aois(aois_raw) -> List[Tuple[str, List[List[float]]]]:
    """
    ## Normalize AOIs to list of (name, polygon[[x,y],...])
    Accepted inputs:
      - [{'name': ..., 'polygon': [[x,y],...]}] or with 'points'/'pts'
      - {'AOI name': [[x,y],...], ...}
      - [[x,y], ...]  -> becomes [("AOI", [[x,y],...])]
      - None / empty -> []
    """
    if not aois_raw:
        return []
    out: List[Tuple[str, List[List[float]]]] = []
    if isinstance(aois_raw, dict):
        if "polygon" in aois_raw or "points" in aois_raw or "pts" in aois_raw:
            poly = aois_raw.get("polygon") or aois_raw.get("points") or aois_raw.get("pts") or []
            out.append((aois_raw.get("name", "AOI 1"), poly))
            return out
        for k, v in aois_raw.items():
            if isinstance(v, dict):
                poly = v.get("polygon") or v.get("points") or v.get("pts")
            else:
                poly = v
            out.append((str(k), poly))
        return out
    if isinstance(aois_raw, list):
        if len(aois_raw) and isinstance(aois_raw[0], (list, tuple)) and len(aois_raw[0]) == 2:
            return [("AOI", aois_raw)]
        for i, it in enumerate(aois_raw, 1):
            if isinstance(it, dict):
                nm = str(it.get("name") or f"AOI {i}")
                poly = it.get("polygon") or it.get("points") or it.get("pts")
                out.append((nm, poly))
            else:
                out.append((f"AOI {i}", it))
        return out
    return out

def build_aoi_masks(h: int, w: int, aois_raw):
    """
    ## Build per-AOI raster masks (uint8, 255=inside)
    Returns list of (name, mask).
    """
    aois = _normalize_aois(aois_raw)
    masks = []
    if not aois:
        return masks
    for nm, poly in aois:
        try:
            pts = np.asarray(poly, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
                continue
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            pts_i = pts.astype(np.int32)
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [pts_i], 255)
            masks.append((nm, m))
        except Exception:
            continue
    return masks

def union_mask_from_masks(h: int, w: int, masks) -> Optional[np.ndarray]:
    """
    ## Build union mask (uint8) from list[(name, mask)]
    """
    if not masks:
        return None
    out = np.zeros((h, w), dtype=np.uint8)
    for _, m in masks:
        out |= (m > 0).astype(np.uint8) * 255
    return out if out.any() else None

def build_union_mask(h: int, w: int, aois_raw) -> Optional[np.ndarray]:
    """
    ## Public wrapper expected by start_app.py
    Creates a union AOI mask (uint8, 0/255) from raw AOI definition.
    """
    masks = build_aoi_masks(h, w, aois_raw)
    return union_mask_from_masks(h, w, masks)

## Back-compat alias (some start_app versions import the plural form)
def build_union_masks(h: int, w: int, aois_raw):
    return build_union_mask(h, w, aois_raw)

def bbox_aoi_overlap_frac(b: Tuple[float,float,float,float], union_mask: np.ndarray) -> float:
    """
    ## Fraction of bbox area that lies within AOI union mask.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    h, w = union_mask.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    sub = union_mask[y1:y2, x1:x2]
    area = float((x2 - x1) * (y2 - y1))
    inside = float((sub > 0).sum())
    return inside / area if area > 0 else 0.0

def point_in_any_polygon(cx: float, cy: float, aois: List[Tuple[str, List[List[float]]]]) -> bool:
    """
    ## Centroid-in-AOI test that does NOT depend on raster masks.
    Uses cv2.pointPolygonTest against each polygon.
    """
    pt = (float(cx), float(cy))
    for _nm, poly in aois:
        if not poly or len(poly) < 3:
            continue
        pts = np.asarray(poly, dtype=np.float32)
        if cv2.pointPolygonTest(pts, pt, False) >= 0:
            return True
    return False


# ------------------------------ device ------------------------------

def select_torch_device(requested: str) -> str:
    """## Choose execution device"""
    req = (requested or "").strip().lower()
    try:
        import torch
        if req in ("cpu", "mps"):
            return req
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


# ------------------------------ runner ------------------------------

def run_legacy_pt(
    imgs,
    outdir: Path,
    model_path: str,
    tile: int, overlap: float,
    conf: float, iou: float,
    selected_classes,
    overlay_mode: str,
    draw_centroid: bool,
    aoi_mode: str,              # 'off' | 'centroid' | 'box' | 'clip'(treated like 'centroid')
    aoi_box_frac: float,
    aoi_map: dict,
    progress_cb,
    stop_cb,
    class_id_to_name: dict | None,
    logger=print,
    return_dets: bool = True,
):
    """
    ## Legacy .pt inference path with robust AOI support
    - AOI overlay always drawn if AOIs exist
    - AOI filtering: 'centroid' (via point-in-polygon), 'box' (via union mask overlap)
    - GIS export: only filtered detections, if image is georeferenced
    - Run-level summaries: detections_full.csv, results_per_image.csv, results_totals.json
    Returns:
      (totals_dict, dets_map) if return_dets is True, else totals_dict
    """
    if YOLO is None:
        raise RuntimeError("Ultralytics not available — install it into local pkgs/_pkgs.")

    outdir = ensure_dir(outdir)
    prv_dir = ensure_dir(outdir / "annotated")

    device = select_torch_device("auto")
    logger(f"[legacy-pt] device={device}")
    model = YOLO(model_path)

    id2name = class_id_to_name or {}
    selected_set = set(selected_classes or [])

    def _progress(pct: float, txt: str = ""):
        if not progress_cb:
            return
        try:
            progress_cb(float(pct), txt)
        except TypeError:
            progress_cb(float(pct))

    def _eta_str(dt_sec: float, frac: float) -> str:
        if frac <= 0 or dt_sec <= 0:
            return ""
        if frac >= 1.0:
            return "done"
        rem = dt_sec * (1.0 - frac) / max(1e-9, frac)
        m, s = divmod(int(rem), 60)
        h, m = divmod(m, 60)
        if h > 0: return f"ETA {h}h{m:02d}m"
        if m > 0: return f"ETA {m}m{s:02d}s"
        return f"ETA {s}s"

    start_t = time()
    totals: Dict[str, int] = {}
    dets_map: Dict[str, List[dict]] = {}

    full_rows: List[List[str]] = []
    per_image_counts: List[Tuple[str, Dict[str, int]]] = []

    N = len(imgs)
    for idx, p in enumerate(imgs, 1):
        if stop_cb and stop_cb():
            logger("[ABORT] user requested stop")
            break

        p = Path(p)
        img = cv2.imread(str(p))
        if img is None:
            logger(f"[WARN] cannot read {p}")
            continue
        H, W = img.shape[:2]

        ## --- build AOI structures for this image ---
        aois: List[Tuple[str, List[List[float]]]] = []
        if aoi_map:
            am = aoi_map.get(str(p)) or aoi_map.get(p.name) or aoi_map.get(str(p.resolve()))
            if am:
                aois = _normalize_aois(am)

        mode = (aoi_mode or "off").lower()
        if mode == "clip":  # legacy UI value; treat like centroid
            mode = "centroid"

        # Build mask ONLY if needed for 'box' mode
        union_mask = None
        if aois and mode == "box":
            masks = build_aoi_masks(H, W, aois)
            union_mask = union_mask_from_masks(H, W, masks)
            if union_mask is None or not union_mask.any():
                logger(f"[AOI] empty union mask for {p.name} (box mode) — AOI will exclude all.")

        logger(f"[AOI] {p.name}: polygons={len(aois)}  mode={mode}  "
               f"mask={'yes' if union_mask is not None else 'no'}")

        ## --- tiling grid ---
        tile = int(tile)
        step = max(1, int(tile * (1.0 - float(overlap))))
        xs = list(range(0, W, step))
        ys = list(range(0, H, step))
        tiles_total = len(xs) * len(ys)
        tiles_done = 0

        all_boxes: List[List[float]] = []
        all_scores: List[float] = []
        all_cids: List[int] = []

        for yy in ys:
            for xx in xs:
                if stop_cb and stop_cb():
                    break
                roi = img[yy:min(yy+tile, H), xx:min(xx+tile, W)]
                res = model.predict(
                    source=roi, imgsz=tile,
                    conf=max(0.05, float(conf)), iou=float(iou),
                    device=device, verbose=False
                )
                if not res:
                    continue
                r = res[0]
                if hasattr(r, "boxes") and r.boxes is not None:
                    b = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else None
                    s = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else None
                    c = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else None
                    if b is None or s is None or c is None:
                        continue
                    if len(b) > 0:
                        b[:, [0,2]] += xx
                        b[:, [1,3]] += yy
                    for bb, ss, cc in zip(b, s, c):
                        all_boxes.append([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])
                        all_scores.append(float(ss))
                        all_cids.append(int(cc))

                tiles_done += 1
                frac_img = tiles_done / max(1, tiles_total)
                frac_all = ((idx - 1) + frac_img) / max(1, N)
                _progress(frac_all * 100.0, _eta_str(time() - start_t, frac_all))

            if stop_cb and stop_cb():
                break

        ## --- NMS + strict conf filter + class filter ---
        keep = _nms_numpy(all_boxes, all_scores, iou_thr=float(iou))
        boxes = [all_boxes[i] for i in keep]
        scores = [all_scores[i] for i in keep]
        cids   = [all_cids[i]   for i in keep]

        sel = []
        for b, s, cid in zip(boxes, scores, cids):
            if s <= float(conf):            # strict '>' like engine-core
                continue
            if selected_set and (cid not in selected_set):
                continue

            ## --- AOI filtering ---
            in_aoi = True
            if aois and mode in ("centroid", "box"):
                if mode == "centroid":
                    cx, cy = bbox_center(b)
                    in_aoi = point_in_any_polygon(cx, cy, aois)
                elif mode == "box":
                    # If mask is missing/empty, exclude (safe default)
                    if union_mask is None or not union_mask.any():
                        in_aoi = False
                    else:
                        frac = bbox_aoi_overlap_frac(b, union_mask)
                        in_aoi = (frac >= float(aoi_box_frac))
            elif mode != "off":
                # AOI is ON in UI, but we have zero polygons for this image -> exclude
                in_aoi = False

            if not in_aoi:
                continue

            sel.append((b, s, cid, in_aoi))

        logger(f"[DEBUG] {p.name}: after NMS {len(keep)}, after conf>{float(conf):.3f} & AOI -> {len(sel)} kept")

        ## --- update totals and per-image counts ---
        cnt: Dict[str, int] = {}
        for _b, _s, cid, _ in sel:
            cname = id2name.get(cid, str(cid))
            totals[cname] = totals.get(cname, 0) + 1
            cnt[cname] = cnt.get(cname, 0) + 1
        per_image_counts.append((p.name, cnt))

        ## --- run-level CSV rows ---
        for (x1,y1,x2,y2), s, cid, in_aoi in sel:
            cx, cy = bbox_center((x1,y1,x2,y2))
            full_rows.append([p.name, id2name.get(cid, str(cid)), f"{s:.6f}",
                              f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                              f"{cx:.2f}", f"{cy:.2f}", 1 if in_aoi else 0])

        ## --- annotated overlay (draw AOIs + per-AOI counts + overall totals) ---
        if overlay_mode and overlay_mode != "off":
            vis = img.copy()

            # 1) Draw AOI polygons
            if aois:
                for nm, poly in aois:
                    if poly and len(poly) >= 3:
                        pts = np.asarray(poly, dtype=np.int32)
                        pts[:,0] = np.clip(pts[:,0], 0, W-1)
                        pts[:,1] = np.clip(pts[:,1], 0, H-1)
                        cv2.polylines(vis, [pts], isClosed=True, color=(255,200,0), thickness=2)

            # 2) Draw kept detections
            for (x1, y1, x2, y2), s, cid, _ in sel:
                color = (0, 255, 0)
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                if draw_centroid:
                    cx, cy = bbox_center((x1, y1, x2, y2))
                    cv2.circle(vis, (int(cx), int(cy)), 3, (255, 255, 255), -1)
                if overlay_mode == "boxes_conf":
                    label = f"{id2name.get(cid, cid)} {s:.2f}"
                    cv2.putText(vis, label, (int(x1), max(0, int(y1)-3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # 3) Per-AOI counts (centroid-in-polygon, independent of current mode)
            if aois:
                aoi_counts = {nm: 0 for nm, _ in aois}
                det_centroids = [bbox_center(b) for (b, _score, _cid, _inaoi) in sel]

                for (nm, poly) in aois:
                    pts = np.asarray(poly, dtype=np.float32)
                    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
                        continue
                    for (cx, cy) in det_centroids:
                        if cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0:
                            aoi_counts[nm] += 1

                # Draw each AOI label near its top-left of polygon bbox
                for (nm, poly) in aois:
                    pts = np.asarray(poly, dtype=np.float32)
                    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
                        continue
                    x_min = float(np.min(pts[:,0])); y_min = float(np.min(pts[:,1]))
                    label = f"{nm}: {aoi_counts.get(nm,0)}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    x = int(np.clip(x_min, 0, W-1)); y = int(np.clip(y_min - 6, 0, H-1))
                    x2b, y2b = min(W-1, x + tw + 10), min(H-1, y + th + 8)
                    cv2.rectangle(vis, (x, max(0,y-th-6)), (x2b, y2b), (0,0,0), -1)
                    cv2.putText(vis, label, (x+5, y+th//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            # 4) Overall per-image totals (bottom-right)
            if cnt:
                lines = [f"{k}: {v}" for k, v in sorted(cnt.items())]
                text = "  ".join(lines)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                h, w = vis.shape[:2]; pad = 8
                x2b, y2b = w - pad, h - pad
                x1b, y1b = max(0, x2b - tw - 2*pad), max(0, y2b - th - 2*pad)
                cv2.rectangle(vis, (x1b, y1b), (x2b, y2b), (0, 0, 0), -1)
                cv2.putText(vis, text, (x1b+pad, y2b-pad),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            out_img = unique_path(prv_dir / f"{p.stem}_annotated.jpg")
            try:
                cv2.imwrite(str(out_img), vis)
            except Exception as e:
                logger(f"[WARN] cannot write annotated image: {e}")

        ## --- return map for potential extra exports ---
        dets_map[str(p)] = [
            {
                "cls": id2name.get(cid, str(cid)),
                "conf": float(s),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "centroid": [float(bbox_center((x1,y1,x2,y2))[0]), float(bbox_center((x1,y1,x2,y2))[1])],
                "in_aoi": bool(in_aoi),
            }
            for (x1,y1,x2,y2), s, cid, in_aoi in sel
        ]

        ## --- GIS export (only filtered detections) ---
        if export_geojson_for_image is not None:
            try:
                dets_for_geo = []
                for (x1,y1,x2,y2), s, cid, in_aoi in sel:
                    cx, cy = bbox_center((x1,y1,x2,y2))
                    dets_for_geo.append({
                        "cls": id2name.get(cid, str(cid)),
                        "conf": float(s),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "centroid": [float(cx), float(cy)],
                    })
                aois_for_geo = [{"name": nm, "polygon": poly} for (nm, poly) in (aois or [])]
                export_geojson_for_image(p, dets_for_geo, aois_for_geo, out_dir=ensure_dir(outdir / "gis"))
            except Exception as e:
                logger(f"[WARN] GIS export failed for {p.name}: {e}")

        frac_all = idx / max(1, N)
        _progress(frac_all * 100.0, _eta_str(time() - start_t, frac_all))

    ## --- write run-level artifacts ---
    if full_rows:
        p_csv = outdir / "detections_full.csv"
        try:
            with open(unique_path(p_csv), "w", encoding="utf-8") as f:
                f.write("image,cls,conf,x1,y1,x2,y2,cx,cy,in_aoi\n")
                for r in full_rows:
                    f.write(",".join(map(str, r)) + "\n")
        except Exception as e:
            logger(f"[WARN] failed writing {p_csv.name}: {e}")

    if per_image_counts:
        p_img = outdir / "results_per_image.csv"
        try:
            all_classes = sorted({k for _, d in per_image_counts for k in d.keys()})
            with open(unique_path(p_img), "w", encoding="utf-8") as f:
                f.write("image," + ",".join(all_classes) + "\n")
                for img_name, d in per_image_counts:
                    f.write(img_name + "," + ",".join(str(d.get(k,0)) for k in all_classes) + "\n")
        except Exception as e:
            logger(f"[WARN] failed writing {p_img.name}: {e}")

    try:
        import json as _json
        with open(unique_path(outdir / "results_totals.json"), "w", encoding="utf-8") as f:
            _json.dump(totals, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger(f"[WARN] failed writing results_totals.json: {e}")

    ## --- return shape compatible with start_app.py ---
    return (totals, dets_map) if return_dets else totals
