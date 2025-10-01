# legacy_pt_runner.py — YOLO .pt runner with robust AOI filtering and overlay
from __future__ import annotations

from pathlib import Path
from time import time
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # handled later

# We do NOT export GIS here (app does it centrally).
export_geojson_for_image = None

# ------------------------------ utils ------------------------------

def ensure_dir(p: Path | str) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def unique_path(p: Path) -> Path:
    if not Path(p).exists(): return Path(p)
    d, stem, suf = Path(p).parent, Path(p).stem, Path(p).suffix
    k = 1
    while True:
        q = d / f"{stem}_{k}{suf}"
        if not q.exists(): return q
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
    if not xyxy: return []
    idxs = np.argsort(np.asarray(scores))[::-1]
    arr = np.asarray(xyxy, dtype=np.float32)
    keep = []
    while len(idxs) > 0:
        i = int(idxs[0]); keep.append(i)
        if len(idxs) == 1: break
        rest = idxs[1:]
        ious = np.array([_iou_xyxy(arr[i], arr[j]) for j in rest], dtype=np.float32)
        idxs = rest[ious <= iou_thr]
    return keep

# --------------------------- abort helper ---------------------------

def _abort_if_needed(stop_cb):
    """## Hard abort: raise KeyboardInterrupt as soon as the UI sets the stop flag."""
    try:
        if stop_cb and stop_cb():
            raise KeyboardInterrupt("ABORT")
    except KeyboardInterrupt:
        raise
    except Exception:
        # If stop_cb itself fails, still abort
        raise KeyboardInterrupt("ABORT")

# --------------------------- AOI helpers ---------------------------

def _normalize_aois(aois_raw) -> List[Tuple[str, List[List[float]]]]:
    if not aois_raw: return []
    out: List[Tuple[str, List[List[float]]]] = []
    if isinstance(aois_raw, dict):
        if "polygon" in aois_raw or "points" in aois_raw or "pts" in aois_raw:
            poly = aois_raw.get("polygon") or aois_raw.get("points") or aois_raw.get("pts") or []
            out.append((aois_raw.get("name", "AOI 1"), poly)); return out
        for k, v in aois_raw.items():
            if isinstance(v, dict): poly = v.get("polygon") or v.get("points") or v.get("pts")
            else: poly = v
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
    aois = _normalize_aois(aois_raw)
    masks = []
    if not aois: return masks
    for nm, poly in aois:
        try:
            pts = np.asarray(poly, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3: continue
            pts[:,0] = np.clip(pts[:,0], 0, w-1)
            pts[:,1] = np.clip(pts[:,1], 0, h-1)
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [pts.astype(np.int32)], 255)
            masks.append((nm, m))
        except Exception:
            continue
    return masks

def union_mask_from_masks(h: int, w: int, masks) -> Optional[np.ndarray]:
    if not masks: return None
    out = np.zeros((h, w), dtype=np.uint8)
    for _, m in masks:
        out |= (m > 0).astype(np.uint8) * 255
    return out if out.any() else None

def build_union_mask(h: int, w: int, aois_raw) -> Optional[np.ndarray]:
    masks = build_aoi_masks(h, w, aois_raw)
    return union_mask_from_masks(h, w, masks)

def build_union_masks(h: int, w: int, aois_raw):
    return build_union_mask(h, w, aois_raw)

def bbox_aoi_overlap_frac(b: Tuple[float,float,float,float], union_mask: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    h, w = union_mask.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1: return 0.0
    sub = union_mask[y1:y2, x1:x2]
    area = float((x2 - x1) * (y2 - y1))
    inside = float((sub > 0).sum())
    return inside / area if area > 0 else 0.0

def point_in_any_polygon(cx: float, cy: float, aois: List[Tuple[str, List[List[float]]]]) -> Optional[str]:
    pt = (float(cx), float(cy))
    for nm, poly in aois:
        if not poly or len(poly) < 3: continue
        pts = np.asarray(poly, dtype=np.float32)
        if cv2.pointPolygonTest(pts, pt, False) >= 0:
            return nm
    return None

def best_overlap_aoi_name(b, masks: List[Tuple[str, np.ndarray]], min_frac: float) -> Optional[str]:
    if not masks: return None
    best_nm, best_frac = None, 0.0
    for nm, m in masks:
        frac = bbox_aoi_overlap_frac(b, m)
        if frac > best_frac:
            best_frac, best_nm = frac, nm
    return best_nm if best_frac >= float(min_frac) else None

# ------------------------------ device ------------------------------

def select_torch_device(requested: str) -> str:
    req = (requested or "").strip().lower()
    try:
        import torch
        if req in ("cpu", "mps"): return req
        if torch.cuda.is_available() and torch.cuda.device_count() > 0: return "cuda"
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
    aoi_mode: str,              # 'off' | 'centroid' | 'box' | 'clip'
    aoi_box_frac: float,
    aoi_map: dict,
    progress_cb,
    stop_cb,
    class_id_to_name: dict | None,
    logger=print,
    return_dets: bool = True,
):
    if YOLO is None:
        raise RuntimeError("Ultralytics not available.")

    outdir = ensure_dir(outdir)
    prv_dir = ensure_dir(outdir / "annotated")

    device = select_torch_device("auto")
    logger(f"[legacy-pt] device={device}")
    model = YOLO(model_path)

    id2name = class_id_to_name or {}
    selected_set = set(selected_classes or [])

    def _progress(pct: float, txt: str = ""):
        if progress_cb:
            try: progress_cb(float(pct), txt)
            except TypeError: progress_cb(float(pct))

    def _eta_str(dt_sec: float, frac: float) -> str:
        if frac <= 0 or dt_sec <= 0: return ""
        if frac >= 1.0: return "done"
        rem = dt_sec * (1.0 - frac) / max(1e-9, frac)
        m, s = divmod(int(rem), 60); h, m = divmod(m, 60)
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
        _abort_if_needed(stop_cb)  ## hard abort at image start
        p = Path(p)

        img = cv2.imread(str(p))
        if img is None:
            logger(f"[WARN] cannot read {p}"); continue
        H, W = img.shape[:2]

        # --- AOIs for this image ---
        aois = []
        if aoi_map:
            am = aoi_map.get(str(p)) or aoi_map.get(p.name) or aoi_map.get(str(p.resolve()))
            if am: aois = _normalize_aois(am)
        mode = (aoi_mode or "off").lower()
        if mode == "clip": mode = "centroid"

        masks = build_aoi_masks(H, W, aois) if (aois and mode == "box") else []
        union_mask = union_mask_from_masks(H, W, masks) if masks else None

        logger(f"[AOI] {p.name}: polygons={len(aois)}  mode={mode}  mask={'yes' if union_mask is not None else 'no'}")

        # --- tiling ---
        step = max(1, int(int(tile) * (1.0 - float(overlap))))
        xs = list(range(0, W, step)); ys = list(range(0, H, step))
        tiles_total = len(xs) * len(ys); tiles_done = 0

        all_boxes: List[List[float]] = []
        all_scores: List[float] = []
        all_cids: List[int] = []

        for yy in ys:
            for xx in xs:
                _abort_if_needed(stop_cb)  ## abort before heavy predict
                roi = img[yy:min(yy+tile, H), xx:min(xx+tile, W)]
                res = model.predict(source=roi, imgsz=tile, conf=max(0.05, float(conf)), iou=float(iou),
                                    device=device, verbose=False)
                if not res: continue
                r = res[0]
                if hasattr(r, "boxes") and r.boxes is not None:
                    b = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else None
                    s = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else None
                    c = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else None
                    if b is None or s is None or c is None: continue
                    if len(b) > 0:
                        b[:,[0,2]] += xx; b[:,[1,3]] += yy
                    for bb, ss, cc in zip(b, s, c):
                        all_boxes.append([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])
                        all_scores.append(float(ss))
                        all_cids.append(int(cc))
                tiles_done += 1
                frac_img = tiles_done / max(1, tiles_total)
                frac_all = ((idx - 1) + frac_img) / max(1, N)
                _progress(frac_all * 100.0, _eta_str(time() - start_t, frac_all))

        # --- NMS + filters ---
        _abort_if_needed(stop_cb)
        keep = _nms_numpy(all_boxes, all_scores, iou_thr=float(iou))
        boxes = [all_boxes[i] for i in keep]
        scores = [all_scores[i] for i in keep]
        cids   = [all_cids[i]   for i in keep]

        sel = []  # (bbox, score, cid, aoi_name)
        for b, s, cid in zip(boxes, scores, cids):
            if s <= float(conf): continue
            if selected_set and (cid not in selected_set): continue
            aoi_name = None
            in_aoi = True
            if aois and mode in ("centroid","box"):
                if mode == "centroid":
                    cx, cy = bbox_center(b)
                    aoi_name = point_in_any_polygon(cx, cy, aois)
                    in_aoi = aoi_name is not None
                else:  # box
                    if union_mask is None or not union_mask.any():
                        in_aoi = False
                    else:
                        if bbox_aoi_overlap_frac(b, union_mask) <= 0.0:
                            in_aoi = False
                        else:
                            aoi_name = best_overlap_aoi_name(b, masks, aoi_box_frac)
                            in_aoi = aoi_name is not None
            elif mode != "off":
                in_aoi = False
            if not in_aoi: continue
            sel.append((b, s, cid, aoi_name))

        logger(f"[DEBUG] {p.name}: after NMS {len(keep)}, after conf>{float(conf):.3f} & AOI -> {len(sel)} kept")

        # --- totals (image-level) ---
        cnt: Dict[str, int] = {}
        for _b, _s, cid, _aoi in sel:
            cname = id2name.get(cid, str(cid))
            totals[cname] = totals.get(cname, 0) + 1
            cnt[cname] = cnt.get(cname, 0) + 1
        per_image_counts.append((p.name, cnt))

        # --- detections_full rows ---
        for (x1,y1,x2,y2), s, cid, aoi_nm in sel:
            cx, cy = bbox_center((x1,y1,x2,y2))
            full_rows.append([p.name, id2name.get(cid, str(cid)), f"{s:.6f}",
                              f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                              f"{cx:.2f}", f"{cy:.2f}", (aoi_nm or "")])

        # ---------------------------------------------------------------------
        # Annotated image: ALWAYS write one, even if overlay_mode == 'off'
        # We draw polylines/boxes only when overlay_mode != 'off', but we
        # always add the bottom-right summary and save the image.
        # ---------------------------------------------------------------------
        vis = img.copy()

        if overlay_mode and overlay_mode != "off":
        # AOI outlines (if any) — drawn in all visual modes
            if aois:
                for nm, poly in aois:
                    if poly and len(poly) >= 3:
                        pts = np.asarray(poly, dtype=np.int32)
                        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
                        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
                        cv2.polylines(vis, [pts], isClosed=True, color=(255, 200, 0), thickness=2)

            # --- visualization modes ---
            mode_viz = (overlay_mode or "").strip().lower()

            if mode_viz in ("boxes", "boxes_conf"):
                # draw rectangles; optional centroid dot if user checked the box
                for (x1, y1, x2, y2), s, cid, _aoi_nm in sel:
                    color = (0, 255, 0)
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    if draw_centroid:
                        cx, cy = bbox_center((x1, y1, x2, y2))
                        cv2.circle(vis, (int(cx), int(cy)), 3, (255, 255, 255), -1)
                    if mode_viz == "boxes_conf":
                        label = f"{id2name.get(cid, cid)} {s:.2f}"
                        cv2.putText(
                            vis, label, (int(x1), max(0, int(y1) - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                        )

            elif mode_viz in ("centroid", "centroids", "dots"):
                # centroid-only: NO rectangles, just a dot per detection (ignore checkbox)
                for (x1, y1, x2, y2), _s, _cid, _aoi_nm in sel:
                    cx, cy = bbox_center((x1, y1, x2, y2))
                    cv2.circle(vis, (int(cx), int(cy)), 3, (255, 255, 255), -1)

        # ---- bottom-right summary (AOI-aware) ----
        total_count = len(sel)

        # Global per-class breakdown
        overall_by_class: Dict[str, int] = {}
        for _b, _s, cid, _aoi_nm in sel:
            cname = id2name.get(cid, str(cid))
            overall_by_class[cname] = overall_by_class.get(cname, 0) + 1

        # AOIs are considered active only if mode is centroid/box AND polygons exist
        aois_active = (mode in ("centroid", "box") and len(aois) > 0)

        if aois_active:
            # Per-AOI breakdown
            aoi_totals: Dict[str, int] = {}
            aoi_by_class: Dict[str, Dict[str, int]] = {}
            for _b, _s, cid, aoi_nm in sel:
                nm = aoi_nm or "AOI"
                aoi_totals[nm] = aoi_totals.get(nm, 0) + 1
                cname = id2name.get(cid, str(cid))
                if nm not in aoi_by_class: aoi_by_class[nm] = {}
                aoi_by_class[nm][cname] = aoi_by_class[nm].get(cname, 0) + 1

            lines = [f"Total: {total_count}"]
            for nm in sorted(aoi_totals.keys()):
                classes_str = ""
                if aoi_by_class.get(nm):
                    cls_parts = [f"{cn} {aoi_by_class[nm][cn]}" for cn in sorted(aoi_by_class[nm].keys())]
                    classes_str = " (" + ", ".join(cls_parts) + ")"
                lines.append(f"{nm}: {aoi_totals[nm]}{classes_str}")
        else:
            # Only total + global per-class breakdown
            classes_str = ""
            if overall_by_class:
                cls_parts = [f"{cn} {overall_by_class[cn]}" for cn in sorted(overall_by_class.keys())]
                classes_str = " (" + ", ".join(cls_parts) + ")"
            lines = [f"Total: {total_count}{classes_str}"]

        text = "   ".join(lines)  # spaces only (no special bullets)

        # draw summary box bottom-right
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        h, w = vis.shape[:2]; pad = 8
        x2b, y2b = w - pad, h - pad
        x1b, y1b = max(0, x2b - tw - 2*pad), max(0, y2b - th - 2*pad)
        cv2.rectangle(vis, (x1b, y1b), (x2b, y2b), (0,0,0), -1)
        cv2.putText(vis, text, (x1b+pad, y2b-pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        out_img = unique_path(prv_dir / f"{p.stem}_annotated.jpg")
        try:
            ok = cv2.imwrite(str(out_img), vis)
            if not ok:
                logger(f"[WARN] OpenCV refused to write annotated image for {p.name}")
        except Exception as e:
            logger(f"[WARN] cannot write annotated image: {e}")

        # --- dets map for GIS export / CSV ---
        dets_map[str(p)] = [
            {
                "cls": id2name.get(cid, str(cid)),
                "conf": float(s),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "centroid": [float(bbox_center((x1,y1,x2,y2))[0]), float(bbox_center((x1,y1,x2,y2))[1])],
                "aoi": (aoi_nm or ""),
            }
            for (x1,y1,x2,y2), s, cid, aoi_nm in sel
        ]

        _abort_if_needed(stop_cb)  ## abort before end-of-image progress
        frac_all = idx / max(1, N)
        _progress(frac_all * 100.0, _eta_str(time() - start_t, frac_all))

    # --- run-level artifacts ---
    if full_rows:
        p_csv = outdir / "detections_full.csv"
        try:
            with open(unique_path(p_csv), "w", encoding="utf-8") as f:
                f.write("image,cls,conf,x1,y1,x2,y2,cx,cy,aoi\n")
                for r in full_rows:
                    f.write(",".join(map(str, r)) + "\n")
        except Exception as e:
            logger(f"[WARN] failed writing {p_csv.name}: {e}")

    if per_image_counts:
        try:
            all_classes = sorted({k for _, d in per_image_counts for k in d.keys()})
            with open(unique_path(outdir / "results_per_image.csv"), "w", encoding="utf-8") as f:
                f.write("image," + ",".join(all_classes) + "\n")
                for img_name, d in per_image_counts:
                    f.write(img_name + "," + ",".join(str(d.get(k,0)) for k in all_classes) + "\n")
        except Exception as e:
            logger(f"[WARN] failed writing results_per_image.csv: {e}")

    try:
        import json as _json
        with open(unique_path(outdir / "results_totals.json"), "w", encoding="utf-8") as f:
            _json.dump(totals, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger(f"[WARN] failed writing results_totals.json: {e}")

    return (totals, dets_map) if return_dets else totals
