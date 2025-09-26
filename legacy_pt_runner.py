# legacy_pt_runner.py — legacy PyTorch tiling path; per-tile progress; boxes_conf shows class+conf; centroid dots; GIS CSV; detections_full.csv
from __future__ import annotations
from geo_export import export_geojson_for_image
import csv, json, time
from pathlib import Path

try:
    import cv2, numpy as np
    from ultralytics import YOLO
except Exception:
    cv2 = None; np = None; YOLO = None

from app_core import save_csv, save_json

def unique_path(path: Path) -> Path:
    if not path.exists(): return path
    stem = path.stem
    base = stem.rsplit("_",1)[0] if "_" in stem and stem.rsplit("_",1)[-1].isdigit() else stem
    i = 2
    while True:
        c = path.with_name(f"{base}_{i}{path.suffix}")
        if not c.exists(): return c
        i += 1

def _iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    inter = w * h
    if inter <= 0: return 0.0
    area_a = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
    union = area_a + area_b - inter
    return inter / union if union>0 else 0.0

def _nms_numpy(xyxy, scores, iou_thr=0.5):
    if len(xyxy)==0: return []
    import numpy as _np
    idxs = _np.argsort(scores)[::-1]; keep = []
    while idxs.size>0:
        i = idxs[0]; keep.append(i)
        if idxs.size==1: break
        rest = idxs[1:]
        ious = _np.array([_iou_xyxy(xyxy[i], xyxy[j]) for j in rest], dtype=_np.float32)
        idxs = rest[ious <= iou_thr]
    return keep

def bbox_center(b): return (0.5*(b[0]+b[2]), 0.5*(b[1]+b[3]))

def select_torch_device(requested: str) -> str:
    req = (requested or "").strip().lower()
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu_n = torch.cuda.device_count() if cuda else 0
    except Exception:
        cuda = False; gpu_n = 0
    if req in ("", "auto", "gpu", "cuda", "0"):
        return "0" if (cuda and gpu_n>0) else "cpu"
    if req in ("cpu", "-1"):
        return "cpu"
    return req

def build_aoi_masks(h, w, aois):
    if cv2 is None or np is None or not aois:
        return []
    masks = []
    for a in aois:
        poly = a.get("polygon") or []
        if len(poly) >= 3:
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [np.array(poly, dtype=np.int32)], 255)
            masks.append((a.get("name","AOI"), m))
    return masks

def build_union_mask(h, w, aois):
    if cv2 is None or np is None or not aois:
        return None
    m = np.zeros((h, w), dtype=np.uint8)
    for a in aois:
        poly = a.get("polygon") or []
        if len(poly) >= 3:
            cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
    return m

def run_legacy_pt(
    imgs, outdir: Path, model_path: str,
    tile: int, overlap: float, conf: float, iou: float,
    selected_classes, overlay_mode: str, draw_centroid: bool,
    aoi_mode: str, aoi_box_frac: float, aoi_map: dict,
    progress_cb, stop_cb, class_id_to_name: dict|None, logger
):
    if cv2 is None or np is None or YOLO is None:
        raise RuntimeError("Legacy PT path requires opencv-python, numpy, ultralytics installed.")
    outdir.mkdir(parents=True, exist_ok=True)
    ann_dir = (outdir / "annotations"); ann_dir.mkdir(exist_ok=True, parents=True)
    prv_dir = (outdir / "annotated");  prv_dir.mkdir(exist_ok=True, parents=True)

    device = select_torch_device("auto")
    logger(f"[legacy-pt] device={device}")
    model_pt = YOLO(model_path)

    # ---- per-run accumulators ----
    totals = {}
    full_rows = []  # rows for detections_full.csv -> [image,cls,conf,x1,y1,x2,y2,cx,cy,in_aoi]

    logger(f"[DEBUG] conf threshold (raw, strict '>'): {float(conf):.6f}; iou={float(iou):.3f}")

    N = len(imgs)
    t0 = time.time()

    for idx, p in enumerate(imgs, 1):
        if stop_cb(): break
        img = cv2.imread(str(p))
        if img is None:
            logger(f"[WARN] cannot read {p}")
            continue
        H, W = img.shape[:2]

        # --- tiling
        step = max(1, int(tile*(1.0-overlap)))
        xs = list(range(0, W, step)); ys = list(range(0, H, step))
        tiles_total = len(xs) * len(ys)
        tiles_done = 0

        all_boxes=[]; all_scores=[]; all_cids=[]

        for yy in ys:
            for xx in xs:
                if stop_cb(): break
                roi = img[yy:min(yy+tile,H), xx:min(xx+tile,W)]
                res = model_pt.predict(
                    source=roi, conf=float(conf), iou=float(iou), imgsz=tile,
                    device=device, classes=selected_classes, verbose=False
                )
                for r in res:
                    if r.boxes is None: continue
                    b = r.boxes.xyxy.cpu().numpy()
                    s = r.boxes.conf.cpu().numpy().astype(float)
                    c = r.boxes.cls.cpu().numpy().astype(int)
                    if b.size == 0: continue
                    b[:, [0,2]] += xx; b[:, [1,3]] += yy  # to global
                    for bb, ss, cc in zip(b, s, c):
                        all_boxes.append([float(bb[0]),float(bb[1]),float(bb[2]),float(bb[3])])
                        all_scores.append(float(ss)); all_cids.append(int(cc))

                tiles_done += 1
                frac_all = ((idx-1) + (tiles_done / max(1, tiles_total))) / max(1, N)
                elapsed = time.time() - t0
                total = (elapsed/frac_all) if frac_all > 1e-6 else 0.0
                remain = max(0.0, total - elapsed)
                m = int(remain // 60); s = int(remain % 60)
                progress_cb(frac_all*100.0, f"Image {idx}/{N} — ETA {m:02d}:{s:02d}")

        # --- NMS then STRICT conf filter (no rounding)
        if all_boxes:
            keep = _nms_numpy(np.array(all_boxes, np.float32), np.array(all_scores, np.float32), iou_thr=float(iou))
            all_boxes = [all_boxes[k] for k in keep]
            all_scores = [all_scores[k] for k in keep]
            all_cids = [all_cids[k] for k in keep]

        pre_conf_count = len(all_scores)
        if pre_conf_count:
            b = np.array(all_boxes, dtype=np.float32)
            s = np.array(all_scores, dtype=np.float32)
            c = np.array(all_cids, dtype=int)
            mask = s > float(conf)  # STRICT '>'
            b, s, c = b[mask], s[mask], c[mask]
            all_boxes = b.tolist(); all_scores = s.tolist(); all_cids = c.tolist()

        logger(f"[DEBUG] {Path(p).name}: after NMS {pre_conf_count}, after conf>{conf:.3f} -> {len(all_scores)} kept")

        # --- AOIs
        aois = (aoi_map.get(str(p)) or [])  # expected [(name, polygon), ...]
        masks = build_aoi_masks(H, W, aois) if aois else []
        union_mask = build_union_mask(H, W, aois) if aois else None

        kept_idx = list(range(len(all_boxes)))
        if aois and union_mask is not None:
            if aoi_mode == "box":
                thr = float(aoi_box_frac or 0.0)
                kidx = []
                for i in kept_idx:
                    x1,y1,x2,y2 = [max(0,int(v)) for v in all_boxes[i]]
                    area = max(1, (x2-x1)*(y2-y1))
                    inter = int((union_mask[y1:y2, x1:x2] > 0).sum())
                    if inter / float(area) >= thr:
                        kidx.append(i)
                kept_idx = kidx
            else:
                kidx = []
                for i in kept_idx:
                    bb = all_boxes[i]
                    cx = (bb[0]+bb[2]) * 0.5
                    cy = (bb[1]+bb[3]) * 0.5
                    ix, iy = int(cx), int(cy)
                    if 0 <= ix < W and 0 <= iy < H and union_mask[iy, ix] > 0:
                        kidx.append(i)
                kept_idx = kidx

        # --- counts
        id2name = class_id_to_name if class_id_to_name else {i:str(i) for i in sorted(set(all_cids))}
        counts_global: dict[str,int] = {}
        for i in kept_idx:
            cname = id2name.get(all_cids[i], str(all_cids[i]))
            counts_global[cname] = counts_global.get(cname, 0) + 1

        # --- add rows for detections_full.csv (AFTER AOI filtering)
        in_aoi_any = 1 if (aois and union_mask is not None) else 0
        for i in kept_idx:
            bb  = all_boxes[i]
            sc  = float(all_scores[i])
            cc  = int(all_cids[i])
            cx  = (bb[0] + bb[2]) * 0.5
            cy  = (bb[1] + bb[3]) * 0.5
            cname = id2name.get(cc, str(cc))
            full_rows.append([
                Path(p).name, cname, f"{sc:.6f}",
                f"{bb[0]:.2f}", f"{bb[1]:.2f}", f"{bb[2]:.2f}", f"{bb[3]:.2f}",
                f"{cx:.2f}", f"{cy:.2f}", in_aoi_any
            ])

        # --- per-image JSON (debug/trace)
        dets_json = []
        for i in kept_idx:
            bb, sc, cc = all_boxes[i], all_scores[i], all_cids[i]
            cx = (bb[0]+bb[2]) * 0.5
            cy = (bb[1]+bb[3]) * 0.5
            dets_json.append({"bbox": bb, "score": float(sc), "class_id": int(cc),
                              "class_name": id2name.get(cc, str(cc)), "cx": cx, "cy": cy})
        with open(ann_dir / f"{Path(p).stem}.json", "w", encoding="utf-8") as f:
            json.dump({"image": str(p), "counts_global": counts_global, "detections": dets_json},
                      f, ensure_ascii=False, indent=2)

        # --- GIS CSV export (points + boxes + AOIs)
        try:
            dets_for_geo = []
            for i in kept_idx:
                bb, sc, cc = all_boxes[i], all_scores[i], all_cids[i]
                cx = (bb[0]+bb[2]) * 0.5
                cy = (bb[1]+bb[3]) * 0.5
                dets_for_geo.append({
                    "cls": id2name.get(cc, str(cc)),
                    "conf": float(sc),
                    "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                    "centroid": [float(cx), float(cy)]
                })
            aois_struct = [{"name": name, "polygon": poly} for (name, poly) in (aois or [])]
            geo_path = export_geojson_for_image(Path(p), dets_for_geo, aois_struct, outdir)
            if geo_path:
                logger(f"[GEO] Wrote {geo_path.name}")
        except Exception as ge:
            logger(f"[GEO][WARN] export failed for {p}: {ge}")

        # --- annotated preview (with bottom-right summary)
        if overlay_mode in ("boxes","boxes_conf","centroid"):
            draw = img.copy()
            # AOIs
            if aois:
                for name, poly in aois:
                    if len(poly) >= 3:
                        pts = np.array(poly, dtype=np.int32)
                        cv2.polylines(draw, [pts], isClosed=True, color=(255, 200, 0), thickness=2)

            # detections
            for i in kept_idx:
                x1,y1,x2,y2 = [int(v) for v in all_boxes[i]]
                cc = int(all_cids[i])
                name = id2name.get(cc, str(cc))
                sc = float(all_scores[i])
                if overlay_mode in ("boxes", "boxes_conf"):
                    cv2.rectangle(draw, (x1,y1), (x2,y2), (0, 220, 0), 2)
                    if overlay_mode == "boxes_conf":
                        label = f"{name} {sc:.2f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(draw, (x1, y1- th - 6), (x1 + tw + 6, y1), (0,220,0), -1)
                        cv2.putText(draw, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
                if overlay_mode == "centroid" or draw_centroid:
                    cx = int((all_boxes[i][0]+all_boxes[i][2]) * 0.5)
                    cy = int((all_boxes[i][1]+all_boxes[i][3]) * 0.5)
                    cv2.circle(draw, (cx, cy), 3, (255,255,255), -1)

            # bottom-right class counts
            if counts_global:
                lines = [f"{k}: {v}" for k,v in sorted(counts_global.items())]
                text = "  ".join(lines)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                h, w = draw.shape[:2]
                pad = 8
                x2, y2 = w - pad, h - pad
                x1, y1 = max(0, x2 - tw - 2*pad), max(0, y2 - th - 2*pad)
                cv2.rectangle(draw, (x1, y1), (x2, y2), (0,0,0), -1)
                cv2.putText(draw, text, (x1+pad, y2-pad), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            cv2.imwrite(str(prv_dir / f"{Path(p).stem}_ann.jpg"), draw)

        # --- totals
        for nm, v in counts_global.items():
            totals[nm] = totals.get(nm, 0) + v

    # write the full detection list CSV (once per run)
    if full_rows:
        hdr = ["image","cls","conf","x1","y1","x2","y2","cx","cy","in_aoi"]
        save_csv(full_rows, header=hdr, out_path=outdir/"detections_full.csv")
        logger(f"[INFO] Wrote {outdir/'detections_full.csv'}")

    return totals
