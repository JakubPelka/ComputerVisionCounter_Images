# app_core.py  — strict conf filter + full detections CSV
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import math, time, datetime
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Dict

from output_utils import (
    DETECTIONS_FULL_CSV,
    RUN_METADATA_JSON,
    ensure_dir as _ensure_dir,
    unique_path,
    write_csv,
    write_json,
)
from project_paths import add_local_package_paths

import numpy as np

from engine_loader import load_yolo_model, resolve_device

try:
    import cv2  # type: ignore
    from PIL import Image, ImageDraw  # for AOI masks
except Exception:
    cv2 = None
    Image = None
    ImageDraw = None

Point = Tuple[float, float]  # (x, y) image-space
Polygon = List[Point]

# -------------------- IO utils --------------------

def ensure_dir(p: Path) -> None:
    _ensure_dir(p)

def collect_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    imgs = [p for p in path.rglob("*") if p.suffix.lower() in exts]
    imgs.sort()
    return imgs

def save_csv(rows: Sequence[Sequence], header: Optional[Sequence[str]], out_path: Path) -> None:
    write_csv(rows, out_path, header=header, unique=True)

def save_json(obj, out_path: Path) -> None:
    write_json(obj, out_path, unique=True)

def safe_path(p: Path) -> Path:
    """Avoid overwriting by appending _1, _2, ..."""
    return unique_path(p)

def common_input_root(paths: List[Path]) -> Path:
    if not paths: return Path.cwd()
    parts = [p.resolve().parts for p in paths]
    prefix = []
    for z in zip(*parts):
        if all(x == z[0] for x in z):
            prefix.append(z[0])
        else:
            break
    return Path(*prefix) if prefix else paths[0].parent

# -------------------- Geometry / AOI --------------------

def area_poly(poly: Polygon) -> float:
    if not poly or len(poly) < 3: return 0.0
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5

def _line_intersection(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-12: return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if 0.0 <= ua <= 1.0 and 0.0 <= ub <= 1.0:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)
    return None

def poly_clip(subject: Polygon, clip: Polygon) -> Polygon:
    """Sutherland–Hodgman polygon clipping."""
    def inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0
    def compute_intersection(p1, p2, a, b):
        return _line_intersection(p1, p2, a, b)

    output = subject[:]
    if len(clip) < 3:
        return []
    for i in range(len(clip)):
        input_list = output
        output = []
        A = clip[i]; B = clip[(i+1) % len(clip)]
        if not input_list: break
        S = input_list[-1]
        for E in input_list:
            if inside(E, A, B):
                if not inside(S, A, B):
                    inter = compute_intersection(S, E, A, B)
                    if inter: output.append(inter)
                output.append(E)
            elif inside(S, A, B):
                inter = compute_intersection(S, E, A, B)
                if inter: output.append(inter)
            S = E
    return output

def point_in_polygon(pt: Point, poly: Polygon) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    if n < 3: return True
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        if ((y1 > y) != (y0 > y)) and (x < (x0 - x1) * (y - y1) / (y0 - y1 + 1e-12) + x1):
            inside = not inside
        x0, y0 = x1, y1
    return inside

def box_center(xyxy: np.ndarray) -> Point:
    x1, y1, x2, y2 = map(float, xyxy)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def bbox_polygon(xyxy: np.ndarray) -> Polygon:
    x1, y1, x2, y2 = map(float, xyxy)
    return [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]

# -------------------- Tiling / WBF / seam-aware dedup --------------------

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0: return 0.0
    area_a = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
    union = area_a + area_b - inter
    return inter / max(union, 1e-9)

def weighted_box_fusion(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                        iou_thr: float = 0.55, alpha: float = 1.0,
                        weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple class-wise WBF: groups boxes by IoU >= iou_thr and same class.
    Fused coords = weighted average; weight = (score**alpha) * (weights or 1).
    Fused score = max(scores in group).
    """
    if len(boxes) == 0:
        return boxes, scores, classes
    out_b, out_s, out_c = [], [], []
    used = np.zeros(len(boxes), dtype=bool)
    eff_w = (scores ** alpha) * (weights if weights is not None else 1.0)
    for ci in np.unique(classes):
        idxs = np.where(classes == ci)[0]
        idxs = idxs[np.argsort(-scores[idxs])]
        for i in idxs:
            if used[i]: continue
            group = [i]
            for j in idxs:
                if used[j] or j == i: continue
                if iou_xyxy(boxes[i], boxes[j]) >= iou_thr:
                    group.append(j)
            used[group] = True
            ws = eff_w[group][:,None]
            fused = (boxes[group] * ws).sum(axis=0) / max(ws.sum(), 1e-9)
            out_b.append(fused.tolist())
            out_s.append(float(scores[group].max()))
            out_c.append(int(ci))
    return np.array(out_b, dtype=float), np.array(out_s, dtype=float), np.array(out_c, dtype=int)

def seam_weights_for_tile_boxes(boxes: np.ndarray, tile_xywh: Tuple[int,int,int,int],
                                band_factor: float = 0.1, weight: float = 0.7) -> np.ndarray:
    """
    Lower weight for detections near tile seams to reduce duplicates across tiles.
    If a box center is within 'band' pixels from tile edge -> multiply weight.
    """
    if len(boxes) == 0:
        return np.ones((0,), dtype=float)
    x, y, w, h = tile_xywh
    band = band_factor * min(w, h)
    weights = np.ones((len(boxes),), dtype=float)
    for i, b in enumerate(boxes):
        cx, cy = box_center(b)
        dist = min(cx - x, y + h - cy, cy - y, x + w - cx)
        if dist < band:
            weights[i] *= weight
    return weights

# -------------------- Inference Engine --------------------

@dataclass
class InferConfig:
    model_path: str
    engine: str = "auto"
    device: str = "auto"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 960
    classes: Optional[List[int]] = None  # indices
    aoi_mode: str = "centroid"  # 'centroid' | 'box'
    aoi_box_frac: float = 0.2
    annotate: bool = True
    draw_centroid: bool = False
    # tiling
    use_tiling: bool = False
    tile: int = 960
    overlap: int = 160
    # WBF/seam
    use_wbf: bool = False
    wbf_iou: float = 0.55
    wbf_alpha: float = 1.0
    seam_band_factor: float = 0.12
    seam_weight: float = 0.7
    # overlay
    overlay_mode: str = "boxes_conf"  # 'boxes' | 'boxes_conf' | 'centroid'
    # AOI persist
    persist_aoi_to_input: bool = True

class ModelEngine:
    def __init__(self, cfg: InferConfig):
        self.cfg = cfg
        self.model = load_yolo_model(cfg.model_path, cfg.engine)
        self.device = resolve_device(cfg.engine, cfg.device)
        # Class names
        try:
            self.class_names = self.model.model.names if hasattr(self.model, "model") else self.model.names
        except Exception:
            self.class_names = {}
        if not isinstance(self.class_names, dict):
            try:
                self.class_names = {i: n for i, n in enumerate(self.class_names)}
            except Exception:
                self.class_names = {}

    # ----- meta -----
    def available_classes(self) -> Dict[int, str]:
        return dict(self.class_names)

    # ----- helpers -----
    def _predict_fullimage(self, img_path: Path):
        res_list = self.model.predict(
            source=str(img_path), conf=self.cfg.conf, iou=self.cfg.iou,
            classes=self.cfg.classes, device=self.device, imgsz=self.cfg.imgsz, verbose=False
        )
        res = res_list[0]
        try:
            boxes = res.boxes.xyxy.detach().cpu().numpy()
            cls = res.boxes.cls.detach().cpu().numpy().astype(int)
            conf = res.boxes.conf.detach().cpu().numpy()
        except Exception:
            boxes = np.zeros((0,4)); cls = np.zeros((0,), dtype=int); conf = np.zeros((0,), dtype=float)
        return boxes, conf, cls

    def _predict_tiled(self, img_path: Path):
        if cv2 is None:
            return np.zeros((0,4)), np.zeros((0,)), np.zeros((0,), dtype=int)
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        tile = int(self.cfg.tile)
        ov = int(self.cfg.overlap)
        step = max(1, tile - ov)
        all_boxes, all_scores, all_cls, all_weights = [], [], [], []

        for y in range(0, h, step):
            for x in range(0, w, step):
                tw = min(tile, w - x)
                th = min(tile, h - y)
                crop = img[y:y+th, x:x+tw]
                res = self.model.predict(source=crop, conf=self.cfg.conf, iou=self.cfg.iou,
                                         classes=self.cfg.classes, device=self.device,
                                         imgsz=self.cfg.imgsz, verbose=False)[0]
                try:
                    bx = res.boxes.xyxy.detach().cpu().numpy()
                    sc = res.boxes.conf.detach().cpu().numpy()
                    cl = res.boxes.cls.detach().cpu().numpy().astype(int)
                except Exception:
                    bx = np.zeros((0,4)); sc = np.zeros((0,)); cl = np.zeros((0,), dtype=int)
                if len(bx):
                    # shift to global coords
                    bx[:, [0,2]] += x
                    bx[:, [1,3]] += y
                    all_boxes.append(bx); all_scores.append(sc); all_cls.append(cl)
                    # seam weights
                    eff_w = seam_weights_for_tile_boxes(bx, (x,y,tw,th), self.cfg.seam_band_factor, self.cfg.seam_weight)
                    all_weights.append(eff_w)

        if not all_boxes:
            return np.zeros((0,4)), np.zeros((0,)), np.zeros((0,), dtype=int)

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        classes = np.concatenate(all_cls, axis=0)
        weights = np.concatenate(all_weights, axis=0) if all_weights else None

        # WBF
        if self.cfg.use_wbf:
            boxes, scores, classes = weighted_box_fusion(
                boxes, scores, classes, iou_thr=float(self.cfg.wbf_iou),
                alpha=float(self.cfg.wbf_alpha), weights=weights
            )
        return boxes, scores, classes

    # -------- STRICT CONF FILTER (raw floats, no rounding) --------
    def _filter_by_conf(self, boxes: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        """Apply a strict confidence threshold using raw float values."""
        if boxes is None or conf is None or cls is None or len(conf) == 0:
            return boxes, conf, cls
        thr = float(self.cfg.conf)
        # Guarantee numeric dtype
        conf = conf.astype(float, copy=False)
        mask = conf >= thr  # raw compare; do NOT round anywhere for logic
        if mask.sum() == len(conf):
            return boxes, conf, cls
        return boxes[mask], conf[mask], cls[mask]

    def _apply_aoi(self, boxes: np.ndarray, cls: np.ndarray, aoi_polys: Optional[List[Polygon]]):
        if not aoi_polys or len(boxes) == 0:
            return np.arange(len(boxes), dtype=int)
        keep = []
        if self.cfg.aoi_mode == "centroid":
            for i, b in enumerate(boxes):
                cx, cy = box_center(b)
                if any(point_in_polygon((cx, cy), aoi) for aoi in aoi_polys if len(aoi) >= 3):
                    keep.append(i)
        else:
            thr = float(self.cfg.aoi_box_frac)
            for i, b in enumerate(boxes):
                bpoly = bbox_polygon(b); barea = max(area_poly(bpoly), 1e-9)
                frac_max = 0.0
                for aoi in aoi_polys:
                    if len(aoi) < 3: continue
                    clip = poly_clip(bpoly, aoi)
                    if not clip: continue
                    frac = area_poly(clip) / barea
                    if frac > frac_max:
                        frac_max = frac
                        if frac_max >= thr: break
                if frac_max >= thr:
                    keep.append(i)
        return np.array(keep, dtype=int)

    def _draw_overlay(self, img_bgr, boxes, cls, conf, aoi_polys, counts):
        if cv2 is None:
            return img_bgr
        draw = img_bgr.copy()
        # AOIs
        if aoi_polys:
            for aoi in aoi_polys:
                if len(aoi) >= 3:
                    pts = np.array(aoi, dtype=np.int32)
                    cv2.polylines(draw, [pts], isClosed=True, color=(255, 200, 0), thickness=2)
        mode = self.cfg.overlay_mode
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = map(int, b)
            c = int(cls[i]) if i < len(cls) else -1
            name = self.class_names.get(c, str(c))
            cf = float(conf[i]) if i < len(conf) else 0.0
            if mode in ("boxes", "boxes_conf"):
                cv2.rectangle(draw, (x1,y1), (x2,y2), (0, 220, 0), 2)
                # display-only formatting
                label = f"{name}" if mode=="boxes" else f"{name} {cf:.2f}"
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(draw, (x1, y1- th - 6), (x1 + tw + 6, y1), (0,220,0), -1)
                cv2.putText(draw, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            if mode in ("centroid",) or (mode in ("boxes","boxes_conf") and self.cfg.draw_centroid):
                cx, cy = box_center(b)
                cv2.circle(draw, (int(cx), int(cy)), 3, (255,255,255), -1)
        # Bottom-right class counts
        if counts:
            lines = [f"{k}: {v}" for k,v in sorted(counts.items())]
            text = "  ".join(lines)
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            h, w = draw.shape[:2]
            pad = 8
            x2, y2 = w - pad, h - pad
            x1, y1 = max(0, x2 - tw - 2*pad), max(0, y2 - th - 2*pad)
            cv2.rectangle(draw, (x1, y1), (x2, y2), (0,0,0), -1)
            cv2.putText(draw, text, (x1+pad, y2-pad), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        return draw

    # ----- public API -----
    def predict_image(self, img_path: Path, aois: Optional[List[Polygon]] = None):
        if self.cfg.use_tiling:
            boxes, conf, cls = self._predict_tiled(img_path)
        else:
            boxes, conf, cls = self._predict_fullimage(img_path)

        # Strict confidence filter AFTER any post-processing (e.g., WBF)
        boxes, conf, cls = self._filter_by_conf(boxes, conf, cls)

        # AOI filter
        keep = self._apply_aoi(boxes, cls, aois)
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
        return boxes, conf, cls

    def summarize_counts(self, cls: np.ndarray) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for c in cls:
            name = self.class_names.get(int(c), str(int(c)))
            counts[name] = counts.get(name, 0) + 1
        return counts

    def save_annotated(self, img_path: Path, boxes, cls, conf, counts, aois, out_path: Path):
        if cv2 is None:
            return
        img = cv2.imread(str(img_path))
        draw = self._draw_overlay(img, boxes, cls, conf, aois, counts)
        ensure_dir(out_path.parent)
        out_path = safe_path(out_path)
        cv2.imwrite(str(out_path), draw)

    def save_dets_json(self, img_path: Path, boxes, cls, conf, counts, aois, out_path: Path):
        data = {
            "image": str(img_path),
            "params": {
                **asdict(self.cfg),
                "device": self.device,
            },
            "classes": self.class_names,
            "detections": [
                {
                    "bbox_xyxy": [float(x) for x in b],
                    "cls_id": int(c),
                    "cls_name": self.class_names.get(int(c), str(int(c))),
                    "conf": float(s),  # raw numeric; consumer can format
                    "centroid_xy": [float(x) for x in box_center(b)]
                }
                for b, c, s in zip(boxes, cls, conf)
            ],
            "counts": counts,
            "aoi": aois or []
        }
        save_json(data, out_path)

    def save_aoi_persistence(self, inputs: List[Path], aoi_map: Dict[str, List[Polygon]]):
        """Save AOIs per image to INPUT/aoi/*.json and INPUT/aoi_masks/*.png under common root (if enabled)."""
        if not self.cfg.persist_aoi_to_input or not inputs or not aoi_map:
            return
        root = common_input_root(inputs)
        aoi_dir = root / "aoi"
        mask_dir = root / "aoi_masks"
        ensure_dir(aoi_dir); ensure_dir(mask_dir)
        for p in inputs:
            polys = aoi_map.get(str(p), [])
            if not polys:
                continue
            # JSON  (use 'aois' with 'polygon' to match newer format)
            jpath = aoi_dir / (p.stem + ".json")
            save_json({"image": p.name, "aois": [{"name":"AOI","polygon": polys}]}, jpath)
            # Mask
            if Image is None:
                continue
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            mask = Image.new("L", img.size, 0)
            drw = ImageDraw.Draw(mask)
            for poly in polys:
                if len(poly) >= 3:
                    drw.polygon(poly, outline=255, fill=255)
            mask_path = mask_dir / (p.stem + ".png")
            mask.save(str(safe_path(mask_path)))

    def predict_batch(self, paths: List[Path], aoi_map: Dict[str, List[Polygon]],
                      outdir: Path, annotate: bool,
                      progress_cb: Optional[Callable[[int, int, float], None]] = None,
                      abort_cb: Optional[Callable[[], bool]] = None,
                      return_dets: bool = False):
        """
        return_dets=True -> also returns a dict[str image_path] -> List[det dicts]
        det dict: {'cls': name, 'conf': float, 'bbox': [x1,y1,x2,y2], 'centroid':[cx,cy]}
        """
        per_image_rows: List[Tuple[str, Dict[str, int]]] = []
        dets_map: Dict[str, List[Dict]] = {} if return_dets else {}
        ann_dir = outdir / "annotated"
        det_dir = outdir / "annotations"
        ensure_dir(ann_dir); ensure_dir(det_dir)

        # NEW: full detection rows for CSV
        full_rows: List[List] = []  # image,cls,conf,x1,y1,x2,y2,cx,cy,in_aoi

        t0 = time.time()
        n = len(paths)
        thr = float(self.cfg.conf)
        print(f"[DEBUG] Using conf threshold (raw): {thr:.6f}")

        for i, p in enumerate(paths, 1):
            if abort_cb and abort_cb():
                break
            aois = aoi_map.get(str(p), None)
            boxes, conf, cls = self.predict_image(p, aois=aois)

            # per-image debug
            if len(conf):
                mn, mx = float(conf.min()), float(conf.max())
                print(f"[DEBUG] {p.name}: kept {len(conf)} dets in AOI after conf-filter; conf range [{mn:.4f},{mx:.4f}]")
            else:
                print(f"[DEBUG] {p.name}: kept 0 dets")

            # rows for full CSV
            for b, c, s in zip(boxes, cls, conf):
                cname = self.class_names.get(int(c), str(int(c)))
                cx, cy = box_center(b)
                in_aoi = 1 if (aois and len(aois) > 0) else 0
                full_rows.append([
                    p.name, cname, f"{float(s):.6f}",
                    f"{float(b[0]):.2f}", f"{float(b[1]):.2f}", f"{float(b[2]):.2f}", f"{float(b[3]):.2f}",
                    f"{float(cx):.2f}", f"{float(cy):.2f}", in_aoi
                ])

            cnt = self.summarize_counts(cls)
            per_image_rows.append((str(p), cnt))

            # JSON with detections (engine-native; already conf-filtered)
            self.save_dets_json(p, boxes, cls, conf, cnt, aois, out_path=det_dir / f"{p.stem}.json")

            # Optional dets_map (for GIS/CSV layer writer outside)
            if return_dets:
                dets = []
                for b, c, s in zip(boxes, cls, conf):
                    cname = self.class_names.get(int(c), str(int(c)))
                    cx, cy = box_center(b)
                    dets.append({"cls": cname, "conf": float(s),
                                 "bbox": [float(x) for x in b], "centroid": [float(cx), float(cy)]})
                dets_map[str(p)] = dets

            if annotate and cv2 is not None:
                self.save_annotated(p, boxes, cls, conf, cnt, aois, out_path=ann_dir / f"{p.stem}_ann.jpg")

            if progress_cb:
                elapsed = time.time() - t0
                eta = (elapsed / i) * (n - i) if i > 0 else 0.0
                progress_cb(i, n, eta)

        # Write the full per-detection CSV once per run
        if full_rows:
            full_csv = outdir / DETECTIONS_FULL_CSV
            write_csv(
                full_rows,
                full_csv,
                header=["image", "cls", "conf", "x1", "y1", "x2", "y2", "cx", "cy", "in_aoi"],
                unique=False,
            )
            print(f"[INFO] Wrote per-detection list: {full_csv}")

        # Aggregate totals
        total: Dict[str, int] = {}
        for _, cnt in per_image_rows:
            for k, v in cnt.items():
                total[k] = total.get(k, 0) + v

        if return_dets:
            return per_image_rows, total, dets_map
        return per_image_rows, total

# ---- Run metadata helpers ----
def build_run_metadata(
    outdir: Path,
    inputs: List[Path],
    cfg: InferConfig,
    totals: Dict[str, int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = {
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "inputs_count": len(inputs),
        "inputs": [str(p) for p in inputs],
        "output_dir": str(outdir),
        "params": asdict(cfg),
        "totals": totals
    }
    if extra:
        meta.update(extra)
    return meta


def save_run_metadata(
    outdir: Path,
    inputs: List[Path],
    cfg: InferConfig,
    totals: Dict[str, int],
    extra: Optional[Dict[str, Any]] = None,
):
    meta = build_run_metadata(outdir, inputs, cfg, totals, extra=extra)
    save_json(meta, outdir / RUN_METADATA_JSON)
