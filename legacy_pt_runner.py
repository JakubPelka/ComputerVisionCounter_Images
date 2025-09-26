# legacy_pt_runner.py — legacy PyTorch tiling path; per-tile progress; boxes_conf shows class+conf; centroid dots; optional dets_map for GeoJSON.
from __future__ import annotations
import time, json
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
    progress_cb, stop_cb, class_id_to_name: dict|None, logger,
    return_dets: bool = False
):
    """
    When return_dets=True, returns (totals, dets_map) where:
      dets_map[image_path] = [{'cls': name, 'conf': float, 'bbox':[x1,y1,x2,y2], 'centroid':[cx,cy]}, ...]
    Otherwise returns totals only (backward compatible).
    """
    if cv2 is None or np is None or YOLO is None:
        raise RuntimeError("Legacy PT path requires opencv-python, numpy, ultralytics installed.")
    outdir.mkdir(parents=True, exist_ok=True)
    ann_dir = (outdir / "annotations"); ann_dir.mkdir(exist_ok=True, parents=True)
    prv_dir = (outdir / "annotated"); prv_dir.mkdir(exist_ok=True, parents=True)

    device = select_torch_device("auto")
    logger(f"[legacy-pt] device={device}")
    model_pt = YOLO(model_path)

    totals = {}
    per_rows_global = []
    per_rows_aoi = []
    all_aoi_names = set()
    all_class_names = set()
    dets_map = {} if return_dets else None

    N = len(imgs)
    t0 = time.time()

    for idx, p in enumerate(imgs, 1):
        if stop_cb(): break
        img = cv2.imread(str(p))
        if img is None: logger(f"[WARN] cannot read {p}"); continue
        H,W = img.shape[:2]

        # tiling (progress per tile)
        step = max(1, int(tile*(1.0-overlap)))
        xs = list(range(0, W, step)); ys = list(range(0, H, step))
        tiles_total = len(xs)*len(ys); tiles_done = 0

        all_boxes=[]; all_scores=[]; all_cids=[]
        for yy in ys:
            for xx in xs:
                if stop_cb(): break
                roi = img[yy:min(yy+tile,H), xx:min(xx+tile,W)]
                res = model_pt.predict(source=roi, conf=max(conf-0.05,0.05),
                                       imgsz=tile, device=device, classes=selected_classes, verbose=False)
                for r in res:
                    if r.boxes is None: continue
                    b = r.boxes.xyxy.cpu().numpy()
                    s = r.boxes.conf.cpu().numpy()
                    c = r.boxes.cls.cpu().numpy().astype(int)
                    if b.size==0: continue
                    b[:,[0,2]] += xx; b[:,[1,3]] += yy
                    for bb,ss,cc in zip(b,s,c):
                        all_boxes.append([float(bb[0]),float(bb[1]),float(bb[2]),float(bb[3])])
                        all_scores.append(float(ss)); all_cids.append(int(cc))
                tiles_done += 1
                frac_all = ((idx-1) + (tiles_done / max(1,tiles_total))) / max(1,N)
                elapsed = time.time()-t0
                total = (elapsed/frac_all) if frac_all>1e-6 else 0.0
                remain = max(0.0, total - elapsed)
                m = int(remain // 60); s = int(remain % 60)
                progress_cb(frac_all*100.0, f"Image {idx}/{N} — ETA {m:02d}:{s:02d}")

        # NMS
        if all_boxes:
            keep = _nms_numpy(np.array(all_boxes,np.float32), np.array(all_scores,np.float32), iou_thr=iou)
            all_boxes = [all_boxes[k] for k in keep]
            all_scores = [all_scores[k] for k in keep]
            all_cids = [all_cids[k] for k in keep]

        # AOI masks
        aois = (aoi_map.get(str(p)) or [])
        masks = build_aoi_masks(H, W, aois) if aois else []
        union_mask = build_union_mask(H, W, aois) if aois else None

        # counting
        id2name = class_id_to_name if class_id_to_name else {i:str(i) for i in sorted(set(all_cids))}
        counts_global = {}
        counts_aoi_tot = {}
        counts_aoi_cls = {}
        for cc in sorted(set(id2name.keys())):
            all_class_names.add(id2name[cc])

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
                    cx,cy = map(int, bbox_center(all_boxes[i]))
                    if 0 <= cx < W and 0 <= cy < H and union_mask[cy, cx] > 0:
                        kidx.append(i)
                kept_idx = kidx

        # det list for this image (for GeoJSON export)
        det_list = []

        for i in kept_idx:
            cid = all_cids[i]
            cname = id2name.get(cid, str(cid))
            counts_global[cname] = counts_global.get(cname, 0) + 1
            if return_dets:
                bb = all_boxes[i]
                cx, cy = bbox_center(bb)
                det_list.append({"cls": cname, "conf": float(all_scores[i]),
                                 "bbox": [float(x) for x in bb], "centroid": [float(cx), float(cy)]})
            if masks:
                for aoi_name, m in masks:
                    hit = False
                    if aoi_mode == "box":
                        thr = float(aoi_box_frac or 0.0)
                        x1,y1,x2,y2 = [max(0,int(v)) for v in all_boxes[i]]
                        area = max(1, (x2-x1)*(y2-y1))
                        inter = int((m[y1:y2, x1:x2] > 0).sum())
                        hit = (inter / float(area)) >= thr
                    else:
                        cx,cy = map(int, bbox_center(all_boxes[i]))
                        hit = (0 <= cx < W and 0 <= cy < H and m[cy, cx] > 0)
                    if hit:
                        counts_aoi_tot[aoi_name] = counts_aoi_tot.get(aoi_name, 0) + 1
                        counts_aoi_cls[(aoi_name, cname)] = counts_aoi_cls.get((aoi_name, cname), 0) + 1
                        all_aoi_names.add(aoi_name)

        if return_dets:
            dets_map[str(p)] = det_list

        for nm, v in counts_global.items():
            totals[nm] = totals.get(nm,0)+v

        # JSON per-image for annotations folder
        dets = []
        for i in kept_idx:
            bb, sc, cc = all_boxes[i], all_scores[i], all_cids[i]
            cx,cy = bbox_center(bb)
            dets.append({"bbox":bb, "score":sc, "class_id":int(cc),
                         "class_name": id2name.get(cc,str(cc)), "cx":cx, "cy":cy})
        with open(ann_dir/f"{p.stem}.json","w",encoding="utf-8") as f:
            json.dump({"image": str(p),
                       "counts_global": counts_global,
                       "counts_aoi_total": counts_aoi_tot,
                       "counts_aoi_by_class": {f"{k[0]}::{k[1]}":v for k,v in counts_aoi_cls.items()},
                       "detections": dets},
                      f, ensure_ascii=False, indent=2)

        # annotated preview (+ AOI); centroid dot if requested or overlay=="centroid"
        preview = img.copy()
        if masks:
            for a in aois:
                poly = a.get("polygon") or []
                if len(poly) >= 3:
                    cv2.polylines(preview, [np.array(poly, np.int32)], True, (0,255,255), 2)
        for i in kept_idx:
            x1,y1,x2,y2 = map(int, all_boxes[i])
            if overlay_mode in ("boxes","boxes_conf"):
                cv2.rectangle(preview,(x1,y1),(x2,y2),(0,255,0),2)
            if overlay_mode == "boxes_conf":
                cname = id2name.get(all_cids[i], str(all_cids[i]))
                cv2.putText(preview, f"{cname} {all_scores[i]:.2f}", (x1,max(14,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            if draw_centroid or overlay_mode == "centroid":
                cx,cy = map(int, bbox_center(all_boxes[i]))
                cv2.circle(preview, (cx,cy), 3, (255,255,255), -1)

        # bottom-right summaries
        W = preview.shape[1]; H = preview.shape[0]
        summary = []
        if counts_global:
            summary.append("GLOBAL: " + "  ".join([f"{k}:{v}" for k,v in sorted(counts_global.items())]))
        if counts_aoi_tot:
            for nm, v in sorted(counts_aoi_tot.items()):
                summary.append(f"{nm}: {v}")
        if summary:
            txt = " | ".join(summary)
            (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            x2,y2 = W-10, H-10
            cv2.rectangle(preview, (x2-tw-12,y2-th-10),(x2,y2),(0,0,0),-1)
            cv2.putText(preview, txt, (x2-tw-8,y2-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2, cv2.LINE_AA)

        out_path = unique_path(prv_dir / f"{p.stem}_annotated.jpg")
        cv2.imwrite(str(out_path), preview)

        # CSV rows
        g_names = sorted(counts_global.keys())
        per_rows_global.append(([ "image_path"]+g_names, [str(p)] + [counts_global.get(n,0) for n in g_names]))
        per_rows_aoi.append({"image": str(p), "aoi_totals": dict(counts_aoi_tot), "aoi_cls": dict(counts_aoi_cls)})

    # totals + CSV
    save_json(totals, out_path=outdir/"results_totals.json")
    if per_rows_global:
        final_names = sorted({n for (hdr,_r) in per_rows_global for n in hdr[1:]})
        rows = []
        for (hdr, r) in per_rows_global:
            path = r[0]; local = dict(zip(hdr[1:], r[1:]))
            rows.append([path] + [local.get(n,0) for n in final_names])
        save_csv(rows, header=["image_path"]+final_names, out_path=outdir/"results_per_image.csv")

    if per_rows_aoi:
        all_aoi_names = set(); all_class_names = set()
        for rec in per_rows_aoi:
            all_aoi_names.update(rec["aoi_totals"].keys())
            all_aoi_names.update({k[0] for k in rec["aoi_cls"].keys()})
            all_class_names.update({k[1] for k in rec["aoi_cls"].keys()})
        aoi_total_cols = [f"AOI:{nm}" for nm in sorted(all_aoi_names)]
        aoi_cls_cols = [f"AOI:{a}::{c}" for a in sorted(all_aoi_names) for c in sorted(all_class_names)]
        header = ["image_path"] + aoi_total_cols + aoi_cls_cols
        rows = []
        for rec in per_rows_aoi:
            imgp = rec["image"]; tot = rec["aoi_totals"]; cls = rec["aoi_cls"]
            row = [imgp] + [tot.get(nm,0) for nm in sorted(all_aoi_names)] + \
                  [cls.get((a,c),0) for a in sorted(all_aoi_names) for c in sorted(all_class_names)]
            rows.append(row)
        save_csv(rows, header=header, out_path=outdir/"results_per_image_by_aoi.csv")

    if return_dets:
        return totals, dets_map
    return totals
