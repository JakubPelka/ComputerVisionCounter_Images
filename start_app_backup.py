import os, json, zipfile, threading, time, sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pandas as pd

import torch
from ultralytics import YOLO
from PIL import Image, ImageTk

try:
    import supervision as sv
except Exception:
    sv = None

# --- Scrollowalny kontener na listę klas ---
class ScrollableFrame(tk.Frame):
    def __init__(self, parent, height=160):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, height=height)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas)
        self.win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # aktualizacja scrollregion oraz dopasowanie szerokości
        def _on_config(_event=None):
            try:
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                self.canvas.itemconfigure(self.win, width=self.canvas.winfo_width())
            except Exception:
                pass
        self.inner.bind("<Configure>", _on_config)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.win, width=self.canvas.winfo_width()))

        # obsługa kółka myszy
        self.inner.bind("<Enter>", lambda e: self._bind_wheel())
        self.inner.bind("<Leave>", lambda e: self._unbind_wheel())

    def _bind_wheel(self):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_wheel(self):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        try:
            if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                self.canvas.yview_scroll(-3, "units")
            else:
                self.canvas.yview_scroll(3, "units")
        except Exception:
            pass


# ===================== KONFIG / STAŁE =====================
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MODEL_DIRNAME = "models"

# Presety jakości (Twoje wartości)
QUALITY_PRESETS = {
    1: {"tile": 640,  "overlap": 0.15, "conf": 0.40, "iou_nms": 0.65, "use_wbf": True},
    2: {"tile": 896,  "overlap": 0.25, "conf": 0.45, "iou_nms": 0.60, "use_wbf": True},
    3: {"tile": 1024, "overlap": 0.30, "conf": 0.50, "iou_nms": 0.55, "use_wbf": True},
    4: {"tile": 1280, "overlap": 0.45, "conf": 0.60, "iou_nms": 0.50, "use_wbf": True},
    5: {"tile": 2560, "overlap": 0.60, "conf": 0.75, "iou_nms": 0.40, "use_wbf": True},  # ULTRA
}
DEFAULT_QUALITY = 5  # domyślnie ULTRA
BORDER_PX = 16
DEFAULT_WBF_ALPHA = 0.20

# Domyślne parametry "seam-aware dedup"
DEFAULT_SEAM_IOU_LOW = 0.30       # próg IoU, od którego zaczynamy rozważać zlewanie
DEFAULT_SEAM_BAND_FACTOR = 0.10   # szerokość strefy szwu = factor * step
DEFAULT_SEAM_WEIGHT = 0.35        # waga odległości od szwu
DEFAULT_MARGIN_WEIGHT = 0.25      # waga marginesu od krawędzi obrazu

# ===================== UTIL =====================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def device_auto_str() -> str:
    return "0" if torch.cuda.is_available() else "cpu"

def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    inter = w * h
    if inter <= 0: return 0.0
    area_a = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def nms_numpy(xyxy, scores, iou_thr=0.5):
    if len(xyxy) == 0: return []
    idxs = np.argsort(scores)[::-1]; keep = []
    while idxs.size > 0:
        i = idxs[0]; keep.append(i)
        if idxs.size == 1: break
        rest = idxs[1:]
        ious = np.array([iou_xyxy(xyxy[i], xyxy[j]) for j in rest], dtype=np.float32)
        idxs = rest[ious <= iou_thr]
    return keep

def bbox_center(b):
    x1,y1,x2,y2 = b; return (0.5*(x1+x2), 0.5*(y1+y2))

def point_in_polygon(x, y, polygon_xy):
    poly = np.array(polygon_xy, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

def score_weight_name(p: Path) -> int:
    name = p.name.lower(); score = 0
    if "yolov8m" in name or "_m" in name: score += 100
    if "yolov8n" in name or "_n" in name: score += 50
    if "640" in name: score += 30
    if "448" in name: score += 20
    try: score += int(p.stat().st_size // (1024*1024))
    except Exception: pass
    return score

def find_best_weights(models_dir: Path) -> Path | None:
    cands = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.zip"))
    if not cands: return None
    cands.sort(key=score_weight_name, reverse=True); return cands[0]

def resolve_weights_to_pt(path: Path, extract_dir: Path) -> Path:
    if path.suffix.lower() == ".pt": return path
    if path.suffix.lower() == ".zip":
        ensure_dir(extract_dir)
        with zipfile.ZipFile(path, "r") as z: z.extractall(extract_dir)
        pts = list(extract_dir.rglob("*.pt"))
        if not pts: raise RuntimeError("I arkivet .zip hittades ingen .pt-fil")
        pts.sort(key=score_weight_name, reverse=True); return pts[0]
    raise ValueError("Välj en .pt eller .zip")

# ---------- zapisy kolizyjne ----------
def numbered_path(path: Path) -> Path:
    """Jeśli path istnieje, zwraca path z sufiksem _1/_2/..."""
    if not path.exists():
        return path
    stem, suf = path.stem, path.suffix
    i = 1
    while True:
        p = path.with_name(f"{stem}_{i}{suf}")
        if not p.exists():
            return p
        i += 1

def save_image_collision(img: np.ndarray, path: Path) -> Path:
    """Nadpisz jeśli możesz, jeśli nie – zapisz z sufiksem _N."""
    try:
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        ok = cv2.imwrite(str(path), img)
        if ok:
            return path
    except Exception:
        pass
    # fallback numerowany
    alt = numbered_path(path)
    cv2.imwrite(str(alt), img)
    return alt

def save_json_collision(obj, path: Path) -> Path:
    try:
        if path.exists():
            try: path.unlink()
            except Exception: pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        alt = numbered_path(path)
        with open(alt, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return alt

def save_csv_collision(df: pd.DataFrame, path: Path) -> Path:
    try:
        if path.exists():
            try: path.unlink()
            except Exception: pass
        df.to_csv(path, index=False, encoding="utf-8")
        return path
    except Exception:
        alt = numbered_path(path)
        df.to_csv(alt, index=False, encoding="utf-8")
        return alt

# ===================== AOI (polygon 3–10 punktów, ZOSTAJE JAK BYŁO) =====================
class AOIEditor(tk.Toplevel):
    def __init__(self, master, image_path, existing_points=None, max_points=10):
        super().__init__(master)
        self.title(f"AOI: {Path(image_path).name} — klicka 3–{max_points} punkter (Backspace ångrar, Enter = OK)")
        self.image_path = image_path
        self.points = [] if existing_points is None else [p[:] for p in existing_points]
        self.max_points = max_points
        self._load_image(); self._build_ui()

    def _load_image(self):
        img_bgr = cv2.imread(self.image_path)
        if img_bgr is None:
            messagebox.showerror("AOI", f"Kan inte läsa bilden:\n{self.image_path}")
            self.destroy(); return
        self.h, self.w = img_bgr.shape[:2]
        max_w, max_h = 1280, 800
        scale = min(1.0, max_w/self.w, max_h/self.h)
        self.scale = scale
        disp_w, disp_h = int(self.w*scale), int(self.h*scale)
        img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
        self.tkimg = ImageTk.PhotoImage(Image.fromarray(img_rgb))

    def _build_ui(self):
        self.canvas = tk.Canvas(self, width=self.tkimg.width(), height=self.tkimg.height(), bg="#222")
        self.canvas.pack(fill="both", expand=True)
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)
        self.canvas.bind("<Button-1>", self.on_click)
        self.bind("<BackSpace>", self.on_backspace)
        self.bind("<Return>", self.on_return)
        btns = tk.Frame(self); btns.pack(fill="x", pady=6)
        tk.Button(btns, text="Rensa", command=self.clear).pack(side="left", padx=4)
        tk.Button(btns, text="OK (Enter)", command=self.on_return).pack(side="left", padx=4)
        self._redraw()

    def img_to_disp(self, x, y): return [x*self.scale, y*self.scale]
    def disp_to_img(self, x, y): return [x/self.scale, y/self.scale]

    def _redraw(self):
        self.canvas.delete("aoi")
        for i, p in enumerate(self.points):
            dx, dy = self.img_to_disp(p[0], p[1])
            self.canvas.create_oval(dx-3, dy-3, dx+3, dy+3, fill="yellow", outline="", tags="aoi")
            if i > 0:
                px, py = self.img_to_disp(self.points[i-1][0], self.points[i-1][1])
                self.canvas.create_line(px, py, dx, dy, fill="yellow", width=2, tags="aoi")
        if len(self.points) >= 3:
            x0,y0 = self.img_to_disp(self.points[0][0], self.points[0][1])
            xn,yn = self.img_to_disp(self.points[-1][0], self.points[-1][1])
            self.canvas.create_line(xn, yn, x0, y0, fill="yellow", width=2, tags="aoi")

    def on_click(self, e):
        if len(self.points) >= self.max_points: return
        xi, yi = self.disp_to_img(e.x, e.y)
        self.points.append([xi, yi]); self._redraw()

    def on_backspace(self, _):
        if self.points: self.points.pop(); self._redraw()

    def on_return(self, _event=None):
        # 0 pkt => hela bilden; 3–max_points => valfri polygon
        if len(self.points) == 0:
            self.points = [[0,0],[self.w,0],[self.w,self.h],[0,self.h]]
        if not (len(self.points) == 0 or 3 <= len(self.points) <= self.max_points):
            messagebox.showwarning("AOI", f"Välj 0 punkter (hela bilden) eller 3–{self.max_points} punkter.")
            return
        self.destroy()

    def clear(self):
        self.points = []; self._redraw()

def save_aoi_json_and_mask(input_dir: Path, img_path: Path, points):
    # ZOSTAJE w INPUT/aoi oraz INPUT/aoi_masks
    aoi_dir = ensure_dir(input_dir / "aoi")
    mask_dir = ensure_dir(input_dir / "aoi_masks")
    with open(aoi_dir / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
        json.dump({"image": img_path.name, "points": points}, f, ensure_ascii=False, indent=2)
    im = cv2.imread(str(img_path)); h, w = im.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.array(points, dtype=np.int32); cv2.fillPoly(mask, [poly], 255)
    cv2.imwrite(str(mask_dir / f"{img_path.stem}.png"), mask)

def load_aoi_points_if_exist(input_dir: Path, img_path: Path):
    aoi_json = input_dir / "aoi" / f"{img_path.stem}.json"
    if aoi_json.exists():
        with open(aoi_json, "r", encoding="utf-8") as f: data = json.load(f)
        pts = data.get("points", None)
        if pts and len(pts) >= 3: return pts
    return None

# ===================== WBF + SEAM DEDUP v3 =====================
def weighted_box_fusion(xyxy, scores, cids, margins_norm, iou_thr=0.55, alpha=0.2):
    if len(xyxy) == 0: return [], [], [], []
    order = np.argsort(scores)[::-1]
    xyxy = np.array(xyxy, dtype=np.float32)[order]
    scores = np.array(scores, dtype=np.float32)[order]
    cids = np.array(cids, dtype=int)[order]
    margins_norm = np.array(margins_norm, dtype=np.float32)[order]

    clusters = []
    for b, s, c, m in zip(xyxy, scores, cids, margins_norm):
        placed = False
        for cl in clusters:
            if cl["class"] != c: continue
            if any(iou_xyxy(b, bb) >= iou_thr for bb in cl["boxes"]):
                w = float(s) * (1.0 + alpha * float(m))
                cl["boxes"].append(b); cl["scores"].append(s); cl["weights"].append(w); cl["margins"].append(m)
                placed = True; break
        if not placed:
            w = float(s) * (1.0 + alpha * float(m))
            clusters.append({"class": c, "boxes": [b], "scores": [s], "weights": [w], "margins": [m]})

    fused_boxes, fused_scores, fused_cids, fused_margins = [], [], [], []
    for cl in clusters:
        W = float(np.sum(cl["weights"]))
        if W <= 1e-6:
            fb = cl["boxes"][0]; fs = float(np.max(cl["scores"])); fm = float(np.max(cl["margins"]))
        else:
            arr = np.stack(cl["boxes"], axis=0)
            wcol = np.array(cl["weights"], dtype=np.float32).reshape(-1, 1)
            fb = np.sum(arr * wcol, axis=0) / W
            fs = float(np.max(cl["scores"]))
            fm = float(np.sum(np.array(cl["margins"], dtype=np.float32) * np.array(cl["weights"], dtype=np.float32)) / W)
        fused_boxes.append(fb.tolist()); fused_scores.append(fs); fused_cids.append(int(cl["class"])); fused_margins.append(fm)
    return fused_boxes, fused_scores, fused_cids, fused_margins

def suppress_near_duplicates_v3(
    boxes, scores, cids, margins, img_w, img_h, step,
    iou_low=DEFAULT_SEAM_IOU_LOW, seam_band_factor=DEFAULT_SEAM_BAND_FACTOR,
    seam_weight=DEFAULT_SEAM_WEIGHT, margin_weight=DEFAULT_MARGIN_WEIGHT
):
    """
    Seam-aware dedup v3:
      • pary tej samej klasy o IoU>=iou_low i bliskich centrach LUB IoU>=0.5
      • preferuj box o większym score + wagi: margin_weight*margines + seam_weight*seam-dist
      • seam-dist 0..1: 0 przy linii siatki kafli, 1 daleko od szwu
    """
    if not boxes: return boxes, scores, cids, margins
    boxes = [list(b) for b in boxes]; scores = list(scores); cids = list(cids)
    margins = list(margins) if margins is not None else [None]*len(boxes)

    lines_x = np.arange(step, img_w, step) if step > 0 else np.array([])
    lines_y = np.arange(step, img_h, step) if step > 0 else np.array([])
    seam_band = max(6, int(seam_band_factor * step)) if step > 0 else 12

    def img_margin_norm(b):
        cx, cy = bbox_center(b)
        m = min(cx, cy, img_w - cx, img_h - cy)
        return float(m / (0.5 * min(img_w, img_h)))

    def seam_dist_norm(b):
        if lines_x.size == 0 and lines_y.size == 0: return 1.0
        cx, cy = bbox_center(b)
        dx = np.min(np.abs(lines_x - cx)) if lines_x.size else 1e9
        dy = np.min(np.abs(lines_y - cy)) if lines_y.size else 1e9
        d = float(min(dx, dy))
        return float(min(1.0, d / (seam_band + 1e-6)))  # 0 blisko szwu, 1 daleko

    keep = [True]*len(boxes)
    for i in range(len(boxes)):
        if not keep[i]: continue
        ci = cids[i]; cxi, cyi = bbox_center(boxes[i])
        for j in range(i+1, len(boxes)):
            if not keep[j] or cids[j] != ci: continue
            iou = iou_xyxy(boxes[i], boxes[j])
            if iou < iou_low: continue
            cxj, cyj = bbox_center(boxes[j])
            centers_close = (abs(cxi - cxj) <= 2*seam_band and abs(cyi - cyj) <= 2*seam_band)
            if not (centers_close or iou >= 0.5):  # nie wygląda na dublet na szwie
                continue

            mi = margins[i] if margins[i] is not None else img_margin_norm(boxes[i])
            mj = margins[j] if margins[j] is not None else img_margin_norm(boxes[j])
            si = scores[i] + margin_weight*mi + seam_weight*seam_dist_norm(boxes[i])
            sj = scores[j] + margin_weight*mj + seam_weight*seam_dist_norm(boxes[j])

            if si >= sj: keep[j] = False
            else: keep[i] = False; break

    idx = [k for k,v in enumerate(keep) if v]
    if not idx: return [], [], [], []
    return [boxes[k] for k in idx], [scores[k] for k in idx], [cids[k] for k in idx], [margins[k] for k in idx]

# ===================== AUTO WBF IoU =====================
def auto_wbf_iou(quality: int, iou_nms: float) -> float:
    if int(quality) == 5:  # ULTRA
        return 0.60
    return max(0.55, float(iou_nms))

# ===================== DETEKCJA (KAFELKI) =====================
def detect_image_tiled(
    model: YOLO, img_bgr: np.ndarray, class_idx_list: list,
    conf_base: float, tile: int, overlap: float, iou_nms: float, device_str: str,
    tiles_dir: Path|None=None, raw_dir: Path|None=None, names_map=None,
    use_wbf: bool=True, wbf_alpha: float=0.2,
    wbf_iou_manual: float|None=None, wbf_iou_auto_val: float|None=None,
    progress_cb=None, abort_event: threading.Event|None=None,
    seam_iou_low: float=DEFAULT_SEAM_IOU_LOW, seam_band_factor: float=DEFAULT_SEAM_BAND_FACTOR,
    seam_weight: float=DEFAULT_SEAM_WEIGHT, margin_weight: float=DEFAULT_MARGIN_WEIGHT
):
    """
    Zwraca: boxes, scores, cids, pre_nms_count, fused_margins
    """
    H, W = img_bgr.shape[:2]
    step = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, W, step))
    ys = list(range(0, H, step))
    total_tiles = len(xs) * len(ys)
    tile_counter = 0

    classes_arg = class_idx_list if class_idx_list else None
    conf_infer = max(conf_base - 0.05, 0.05)

    boxes, scores, cids, margins_norm = [], [], [], []
    pre_nms_count = 0; tile_id = 0
    aborted = False

    for y0 in ys:
        for x0 in xs:
            if abort_event is not None and abort_event.is_set():
                aborted = True; break

            x1 = min(x0 + tile, W); y1 = min(y0 + tile, H)
            roi = img_bgr[y0:y1, x0:x1]
            tw, th = (x1 - x0), (y1 - y0)

            if tiles_dir is not None:
                ensure_dir(tiles_dir)
                cv2.imwrite(str(tiles_dir / f"tile_{tile_id:06d}_{x0}_{y0}.jpg"), roi)

            res = model.predict(source=roi, conf=conf_infer, imgsz=tile, device=device_str,
                                classes=classes_arg, verbose=False)

            raw_list = []
            for r in res:
                if r.boxes is None: continue
                bxyxy_local = r.boxes.xyxy.cpu().numpy()
                bconf  = r.boxes.conf.cpu().numpy()
                bcls   = r.boxes.cls.cpu().numpy().astype(int)
                pre_nms_count += len(bxyxy_local)

                if bxyxy_local.size != 0:
                    cx = 0.5 * (bxyxy_local[:,0] + bxyxy_local[:,2])
                    cy = 0.5 * (bxyxy_local[:,1] + bxyxy_local[:,3])
                    m = np.minimum.reduce([cx, cy, (tw - cx), (th - cy)])
                    m_norm = m / (max(1.0, min(tw, th) / 2.0))

                    bxyxy_global = bxyxy_local.copy()
                    bxyxy_global[:, [0,2]] += x0; bxyxy_global[:, [1,3]] += y0

                    for b, s, c, mn in zip(bxyxy_global, bconf, bcls, m_norm):
                        boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                        scores.append(float(s)); cids.append(int(c))
                        margins_norm.append(float(np.clip(mn, 0.0, 1.0)))

                        nm = names_map[c] if names_map is not None else str(c)
                        raw_list.append({"xyxy":[float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                                         "score": float(s), "cid": int(c), "name": nm})

            if raw_dir is not None:
                ensure_dir(raw_dir)
                with open(raw_dir / f"raw_{tile_id:06d}_{x0}_{y0}.json", "w", encoding="utf-8") as f:
                    json.dump(raw_list, f, ensure_ascii=False, indent=2)
            tile_id += 1

            tile_counter += 1
            if progress_cb is not None:
                try:
                    progress_cb(tile_counter, total_tiles)
                except Exception:
                    pass

        if aborted: break

    if len(boxes) == 0:
        return [], [], [], pre_nms_count, []

    fused_margins = None
    if use_wbf:
        if wbf_iou_manual is not None: thr = float(wbf_iou_manual)
        elif wbf_iou_auto_val is not None: thr = float(wbf_iou_auto_val)
        else: thr = max(0.55, iou_nms)
        boxes, scores, cids, fused_margins = weighted_box_fusion(
            boxes, scores, cids, margins_norm, iou_thr=thr, alpha=wbf_alpha
        )

    arr_b = np.array(boxes, dtype=np.float32); arr_s = np.array(scores, dtype=np.float32)
    keep = nms_numpy(arr_b, arr_s, iou_thr=iou_nms)
    boxes = arr_b[keep].tolist(); scores = arr_s[keep].tolist(); cids = (np.array(cids, dtype=int)[keep]).tolist()
    if fused_margins is not None:
        fused_margins = (np.array(fused_margins, dtype=np.float32)[keep]).tolist()

    # Seam-aware dedup v3 (siatka kafli)
    boxes, scores, cids, fused_margins = suppress_near_duplicates_v3(
        boxes, scores, cids, fused_margins if fused_margins is not None else [],
        img_w=W, img_h=H, step=step,
        iou_low=seam_iou_low, seam_band_factor=seam_band_factor,
        seam_weight=seam_weight, margin_weight=margin_weight
    )

    return boxes, scores, cids, pre_nms_count, (fused_margins if fused_margins is not None else [])

# ===================== APP (GUI po szwedzku) =====================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ComputerVision Counter — Count anything without coding")
        self.geometry("1080x880")

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        models_default = Path(__file__).parent / MODEL_DIRNAME
        self.weights_path = tk.StringVar(value=str(find_best_weights(models_default) or models_default))

        self.quality = tk.IntVar(value=DEFAULT_QUALITY)
        self.use_aoi = tk.BooleanVar(value=False)

        # NOWE: tryb nakładki (overlay)
        # 'centroid' | 'boxes' | 'boxes_conf'
        self.overlay_mode = tk.StringVar(value="centroid")

        self.model = None; self.names = None; self.class_vars = []
        self.advanced_override = False
        self.advanced_params = {
            "tile": QUALITY_PRESETS[DEFAULT_QUALITY]["tile"],
            "overlap": QUALITY_PRESETS[DEFAULT_QUALITY]["overlap"],
            "conf": QUALITY_PRESETS[DEFAULT_QUALITY]["conf"],
            "iou_nms": QUALITY_PRESETS[DEFAULT_QUALITY]["iou_nms"],
            "use_wbf": True,
            "wbf_alpha": DEFAULT_WBF_ALPHA,
            "wbf_iou": None,   # None => auto
            "wbf_auto": True,
            # seam-dedup tuning:
            "seam_iou_low": DEFAULT_SEAM_IOU_LOW,
            "seam_band_factor": DEFAULT_SEAM_BAND_FACTOR,
            "seam_weight": DEFAULT_SEAM_WEIGHT,
            "margin_weight": DEFAULT_MARGIN_WEIGHT,
        }
        self.selected_files = []

        # Progress / abort + wątek
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label = tk.StringVar(value="Klar.")
        self.abort_event = threading.Event()
        self.worker_done = threading.Event()
        self.worker_thread = None

        self._build_ui()
        self._autoload_best_model()

    # ---------- UI ----------
    def _build_ui(self):
        frm = tk.Frame(self); frm.pack(fill="both", expand=True, padx=10, pady=10)

        self._row_browse(frm, "Mapp med bilder (in):", self.input_dir, self.browse_input)
        files_row = tk.Frame(frm); files_row.pack(fill="x", pady=3)
        tk.Button(files_row, text="Välj enskilda filer (valfritt)…", command=self.browse_files).pack(side="left")
        self.files_label = tk.Label(files_row, text="— inga valda filer —"); self.files_label.pack(side="left", padx=8)
        tk.Button(files_row, text="Rensa filval", command=self.clear_files).pack(side="left", padx=8)

        self._row_browse(frm, "Utdatamapp (valfritt):", self.output_dir, self.browse_output)
        tk.Label(frm, text="Resultat sparas i undermappen 'results/'. "
                           "Om du inte väljer utdatamapp används '<bildmapp>/results/'.").pack(anchor="w")

        self._row_browse(frm, "Vikter (.pt/.zip):", self.weights_path, self.browse_weights, is_dir=False)

        qf = tk.LabelFrame(frm, text="Kvalitet (1 = snabbare/sämre, 5 = ULTRA)")
        qf.pack(fill="x", pady=6)
        sc = tk.Scale(qf, from_=1, to=5, orient="horizontal", variable=self.quality, showvalue=True,
                      command=lambda _=None: self._update_preset_label())
        sc.pack(side="left", fill="x", expand=True, padx=6)
        self.preset_label = tk.Label(qf, text=""); self.preset_label.pack(side="left", padx=6)
        self._update_preset_label()

        a = tk.LabelFrame(frm, text="AOI (analysområde)")
        a.pack(fill="x", pady=6)
        tk.Checkbutton(a, text="Använd AOI (Enter = hela bilden; rita 3–10 punkter)", variable=self.use_aoi).pack(anchor="w")

        # NOWE: tryb wizualizacji
        vis = tk.LabelFrame(frm, text="Visning (annotering)")
        vis.pack(fill="x", pady=6)
        tk.Radiobutton(vis, text="Centroid (punkter)", variable=self.overlay_mode, value="centroid").pack(side="left", padx=6)
        tk.Radiobutton(vis, text="Rutor utan conf", variable=self.overlay_mode, value="boxes").pack(side="left", padx=6)
        tk.Radiobutton(vis, text="Rutor + conf", variable=self.overlay_mode, value="boxes_conf").pack(side="left", padx=6)

        self.class_frame = tk.LabelFrame(frm, text="Välj klasser (efter inläsning av vikter)")
        self.class_frame.pack(fill="both", expand=True, pady=6)
        self.class_scroll = ScrollableFrame(self.class_frame, height=160)
        self.class_scroll.pack(fill="both", expand=True)
        self.classes_container = self.class_scroll.inner

        act = tk.Frame(frm); act.pack(fill="x", pady=8)
        self.btn_start = tk.Button(act, text="STARTA (kör)", command=self.start)
        self.btn_start.pack(side="left")
        tk.Button(act, text="Avancerade inställningar…", command=self.open_advanced).pack(side="left", padx=8)
        self.btn_abort = tk.Button(act, text="AVBRYT", command=self.abort, state="disabled")
        self.btn_abort.pack(side="left", padx=10)

        # Log + progress
        self.log = tk.Text(frm, height=14); self.log.pack(fill="both", expand=True, pady=(6,2))

        pf = tk.Frame(frm); pf.pack(fill="x", pady=4)
        self.progressbar = ttk.Progressbar(pf, maximum=100.0, variable=self.progress_var)
        self.progressbar.pack(fill="x")
        tk.Label(pf, textvariable=self.progress_label, anchor="w").pack(fill="x")

    def _update_preset_label(self):
        p = QUALITY_PRESETS.get(int(self.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY])
        self.preset_label.config(text=f"tile={p['tile']}  overlap={p['overlap']}  conf={p['conf']}  nms={p['iou_nms']}  WBF={p['use_wbf']}")

    def _row_browse(self, parent, label, var, cmd, is_dir=True):
        f = tk.Frame(parent); f.pack(fill="x", pady=3)
        tk.Label(f, text=label, width=24, anchor="w").pack(side="left")
        tk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(f, text="Välj…", command=cmd).pack(side="left")

    # ---------- Advanced ----------
    def open_advanced(self):
        win = tk.Toplevel(self); win.title("Avancerade inställningar"); win.geometry("660x600")

        tk.Label(win, text="Aktuell preset (enligt reglaget):").pack(anchor="w", padx=8, pady=(8,2))
        p = QUALITY_PRESETS.get(int(self.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY])
        preset_txt = tk.StringVar(value=f"tile={p['tile']}  overlap={p['overlap']}  conf={p['conf']}  nms={p['iou_nms']}  WBF={p['use_wbf']}")
        tk.Entry(win, textvariable=preset_txt, state="readonly").pack(fill="x", padx=8)

        current_auto_wbf = auto_wbf_iou(int(self.quality.get()), float(p["iou_nms"]))

        tk.Label(win, text="Override (lämna tomt = använd preset/auto):").pack(anchor="w", padx=8, pady=(10,2))
        def row(lbltxt, var):
            f = tk.Frame(win); f.pack(fill="x", pady=3)
            tk.Label(f, text=lbltxt, width=30, anchor="w").pack(side="left")
            entry = tk.Entry(f, textvariable=var, width=18); entry.pack(side="left"); return entry

        base = self.advanced_params if self.advanced_override else {
            **p, "wbf_alpha": DEFAULT_WBF_ALPHA, "wbf_iou": None, "wbf_auto": True,
            "seam_iou_low": DEFAULT_SEAM_IOU_LOW, "seam_band_factor": DEFAULT_SEAM_BAND_FACTOR,
            "seam_weight": DEFAULT_SEAM_WEIGHT, "margin_weight": DEFAULT_MARGIN_WEIGHT
        }

        var_tile = tk.StringVar(value=str(base.get("tile","")))
        var_ov   = tk.StringVar(value=str(base.get("overlap","")))
        var_conf = tk.StringVar(value=str(base.get("conf","")))
        var_nms  = tk.StringVar(value=str(base.get("iou_nms","")))
        var_wbf  = tk.BooleanVar(value=bool(base.get("use_wbf", True)))
        var_alpha= tk.StringVar(value=str(base.get("wbf_alpha", DEFAULT_WBF_ALPHA)))
        var_wbf_auto = tk.BooleanVar(value=bool(base.get("wbf_auto", True)))
        var_wbf_iou  = tk.StringVar(value=str(current_auto_wbf) if var_wbf_auto.get() else ("" if base.get("wbf_iou", None) is None else str(base["wbf_iou"])))
        var_seam_iou_low    = tk.StringVar(value=str(base.get("seam_iou_low", DEFAULT_SEAM_IOU_LOW)))
        var_seam_band_fact  = tk.StringVar(value=str(base.get("seam_band_factor", DEFAULT_SEAM_BAND_FACTOR)))
        var_seam_weight     = tk.StringVar(value=str(base.get("seam_weight", DEFAULT_SEAM_WEIGHT)))
        var_margin_weight   = tk.StringVar(value=str(base.get("margin_weight", DEFAULT_MARGIN_WEIGHT)))

        row("Kakelstorlek (px)", var_tile)
        row("Överlapp (0..1)", var_ov)
        row("Konfidens", var_conf)
        row("IoU NMS", var_nms)

        f2 = tk.Frame(win); f2.pack(fill="x", pady=4)
        tk.Checkbutton(f2, text="Använd WBF-dedup", variable=var_wbf).pack(anchor="w", padx=8)

        f3 = tk.Frame(win); f3.pack(fill="x", pady=3)
        tk.Label(f3, text="WBF alpha", width=30, anchor="w").pack(side="left", padx=8)
        tk.Entry(f3, textvariable=var_alpha, width=18).pack(side="left")

        f4 = tk.Frame(win); f4.pack(fill="x", pady=3)
        cb = tk.Checkbutton(f4, text="Auto WBF IoU (ULTRA=0.60, annars=max(0.55, NMS))", variable=var_wbf_auto, command=lambda: toggle_wbf_iou_state())
        cb.pack(anchor="w", padx=8)
        tk.Label(f4, text="WBF IoU (när Auto AV):", width=30, anchor="w").pack(side="left", padx=8)
        ent_wbf = tk.Entry(f4, textvariable=var_wbf_iou, width=18); ent_wbf.pack(side="left")

        row("Skarv-dedup: IoU låg", var_seam_iou_low)
        row("Skarv-zon faktor (× steg)", var_seam_band_fact)
        row("Vikt marginal", var_margin_weight)
        row("Vikt skarv-avstånd", var_seam_weight)

        def toggle_wbf_iou_state():
            if var_wbf_auto.get():
                var_wbf_iou.set(str(current_auto_wbf)); ent_wbf.config(state="disabled")
            else:
                var_wbf_iou.set(""); ent_wbf.config(state="normal")
        toggle_wbf_iou_state()

        btns = tk.Frame(win); btns.pack(fill="x", pady=12)
        def apply_override():
            try:
                cur = QUALITY_PRESETS.get(int(self.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY])
                def get_or(v, cast, key):
                    s = v.get().strip(); return cast(s) if s!="" else cast(cur[key])
                self.advanced_params = {
                    "tile": get_or(var_tile, int, "tile"),
                    "overlap": get_or(var_ov, float, "overlap"),
                    "conf": get_or(var_conf, float, "conf"),
                    "iou_nms": get_or(var_nms, float, "iou_nms"),
                    "use_wbf": bool(var_wbf.get()),
                    "wbf_alpha": float(var_alpha.get()) if var_alpha.get().strip()!="" else DEFAULT_WBF_ALPHA,
                    "wbf_iou": (None if var_wbf_auto.get() else (float(var_wbf_iou.get()) if var_wbf_iou.get().strip()!="" else None)),
                    "wbf_auto": bool(var_wbf_auto.get()),
                    "seam_iou_low": float(var_seam_iou_low.get()) if var_seam_iou_low.get().strip()!="" else DEFAULT_SEAM_IOU_LOW,
                    "seam_band_factor": float(var_seam_band_fact.get()) if var_seam_band_fact.get().strip()!="" else DEFAULT_SEAM_BAND_FACTOR,
                    "seam_weight": float(var_seam_weight.get()) if var_seam_weight.get().strip()!="" else DEFAULT_SEAM_WEIGHT,
                    "margin_weight": float(var_margin_weight.get()) if var_margin_weight.get().strip()!="" else DEFAULT_MARGIN_WEIGHT,
                }
                self.advanced_override = True
                self._log(f"[ADV] Override aktiverad. Auto WBF IoU = {self.advanced_params['wbf_auto']}.")
                win.destroy()
            except Exception as e:
                messagebox.showerror("Avancerat", str(e))

        def reset_to_preset():
            self.advanced_override = False; self._log("[ADV] Återställd preset från reglaget."); win.destroy()

        def apply_anti_seam():
            cur = QUALITY_PRESETS.get(int(self.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY])
            self.advanced_params = {
                "tile": int(var_tile.get()) if var_tile.get().strip() else int(cur["tile"]),
                "overlap": 0.55,
                "conf": float(var_conf.get()) if var_conf.get().strip() else float(cur["conf"]),
                "iou_nms": float(var_nms.get()) if var_nms.get().strip() else float(cur["iou_nms"]),
                "use_wbf": True,
                "wbf_alpha": 0.35,
                "wbf_iou": 0.60,
                "wbf_auto": False,
                "seam_iou_low": 0.40,
                "seam_band_factor": 0.12,
                "seam_weight": 0.45,
                "margin_weight": 0.30,
            }
            self.advanced_override = True
            self._log("[ADV] Anti-seam preset tillämpad.")
            win.destroy()

        tk.Button(btns, text="Tillämpa override", command=apply_override).pack(side="left", padx=6)
        tk.Button(btns, text="Återställ preset", command=reset_to_preset).pack(side="left", padx=6)
        tk.Button(btns, text="Anti-seam (dedup)", command=apply_anti_seam).pack(side="left", padx=6)

    # ---------- File pickers ----------
    def browse_input(self):
        d = filedialog.askdirectory(title="Välj bildmapp")
        if d: self.input_dir.set(d)

    def browse_files(self):
        files = filedialog.askopenfilenames(title="Välj filer",
                                            filetypes=[("Bilder","*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if files:
            self.selected_files = list(files); self.files_label.config(text=f"Valt {len(self.selected_files)} filer")
        else:
            self.selected_files = []; self.files_label.config(text="— inga valda filer —")

    def clear_files(self):
        self.selected_files = []; self.files_label.config(text="— inga valda filer —")

    def browse_output(self):
        d = filedialog.askdirectory(title="Välj utdatamapp")
        if d: self.output_dir.set(d)

    def browse_weights(self):
        initdir = str(Path(self.weights_path.get()).parent) if self.weights_path.get() else str(Path(__file__).parent / MODEL_DIRNAME)
        f = filedialog.askopenfilename(initialdir=initdir, title="Välj vikter",
                                       filetypes=[("Vikter",".pt .zip"), ("Alla filer","*.*")])
        if f:
            self.weights_path.set(f); self.load_model_and_classes()

    def _autoload_best_model(self):
        try:
            wp = self.weights_path.get().strip()
            if not wp or Path(wp).is_dir():
                best = find_best_weights(Path(wp) if wp else (Path(__file__).parent / MODEL_DIRNAME))
                if best: self.weights_path.set(str(best))
            if self.weights_path.get(): self.load_model_and_classes()
        except Exception: pass

    def load_model_and_classes(self):
        try:
            out_base = Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else None
            out_dir = ensure_dir((out_base or Path.cwd()) / "results")
            temp_root = ensure_dir(out_dir / "temp"); extract_dir = ensure_dir(temp_root / "extracted_models")

            wp = Path(self.weights_path.get().strip())
            if wp.is_dir():
                best = find_best_weights(wp)
                if not best: raise FileNotFoundError(f"I {wp} finns inga .pt/.zip")
                wp = best

            pt = resolve_weights_to_pt(wp, extract_dir)
            self._log(f"Läser modell: {pt}")
            self.model = YOLO(str(pt)); self.names = self.model.names
            self._populate_classes(self.names); self._log("Vikter och klasslista inlästa.")
        except Exception as e:
            messagebox.showerror("Modell", f"Kunde inte läsa vikter:\n{e}")

    def _populate_classes(self, names):
        for w in self.classes_container.winfo_children(): w.destroy()
        self.class_vars.clear()
        id2name = list(names.values()) if isinstance(names, dict) else list(names)
        grid = tk.Frame(self.classes_container); grid.pack(fill="both", expand=True)
        cols = 6
        for i, nm in enumerate(id2name):
            var = tk.BooleanVar(value=False)  # domyślnie NIC nie zaznaczamy
            cb = tk.Checkbutton(grid, text=nm, variable=var)
            r, c = divmod(i, cols); cb.grid(row=r, column=c, sticky="w", padx=6, pady=4)
            self.class_vars.append((nm, var, i))

    def selected_class_indices(self):
        return [idx for (nm, v, idx) in self.class_vars if v.get()]

    # ---------- ABORT (natychmiastowy reset) ----------
    def abort(self):
        self.abort_event.set()
        self._set_progress(None, "Avbryter…")
        def _wait_and_reset():
            try:
                if self.worker_thread is not None:
                    self.worker_done.wait(timeout=3.0)
            finally:
                self.after(0, lambda: (
                    self.progress_var.set(0.0),
                    self.progress_label.set("Avbrutet. Klar."),
                    self.btn_start.config(state="normal"),
                    self.btn_abort.config(state="disabled")
                ))
        threading.Thread(target=_wait_and_reset, daemon=True).start()

    # ---------- Start / Run ----------
    def start(self):
        try:
            if self.btn_start['state'] == "disabled":
                return
            self.abort_event.clear()
            self.btn_start.config(state="disabled")
            self.btn_abort.config(state="normal")
            self._set_progress(0.0, "Förbereder…")

            images = []
            if self.selected_files:
                images = [Path(p) for p in self.selected_files]; inp = images[0].parent
            else:
                inp = Path(self.input_dir.get().strip())
                if not inp.exists():
                    messagebox.showerror("In", "Välj en giltig bildmapp eller filer.")
                    self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled")
                    return
                images = sorted([p for p in inp.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
                if not images:
                    self._log("Inga bilder i inmatningsmappen.")
                    self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled")
                    return

            if self.model is None:
                self.load_model_and_classes()
                if self.model is None:
                    self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled")
                    return

            selected_idx = self.selected_class_indices()
            if not selected_idx:
                messagebox.showwarning("Klasser", "Välj minst en klass innan du startar.")
                self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled")
                return

            out_base = Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else (images[0].parent if self.selected_files else inp)
            outp = ensure_dir(out_base / "results")

            aoi_enabled = self.use_aoi.get()
            if aoi_enabled and len(images) > 1:
                self._log("AOI på — kontrollerar/ritar AOI för varje bild…")
                for img_path in images:
                    if self.abort_event.is_set(): break
                    pts = load_aoi_points_if_exist(inp if not self.selected_files else images[0].parent, img_path)
                    if pts is None:
                        ed = AOIEditor(self, str(img_path)); self.wait_window(ed)
                        im = cv2.imread(str(img_path)); h,w = im.shape[:2]
                        pts = ed.points if len(ed.points)>=3 else [[0,0],[w,0],[w,h],[0,h]]
                        save_aoi_json_and_mask(inp if not self.selected_files else images[0].parent, img_path, pts)
                        self._log(f"AOI sparad för {img_path.name}")

            self.worker_done.clear()
            self.worker_thread = threading.Thread(target=self._run, args=(images, outp, aoi_enabled, (images[0].parent if self.selected_files else inp)), daemon=True)
            self.worker_thread.start()
        except Exception as e:
            self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled")
            messagebox.showerror("Fel", str(e))

    def _run(self, images, outp: Path, aoi_enabled: bool, aoi_base_folder: Path):
        t0 = time.time()
        try:
            temp_root = ensure_dir(outp / "temp")
            tiles_root = ensure_dir(temp_root / "tiles")
            raw_root = ensure_dir(temp_root / "raw_preds")
            ann_dir = ensure_dir(outp / "annotations")
            prv_dir = ensure_dir(outp / "previews")

            if self.advanced_override: preset = self.advanced_params.copy()
            else: preset = QUALITY_PRESETS.get(int(self.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY]).copy()

            tile = int(preset["tile"]); overlap = float(preset["overlap"])
            conf = float(preset["conf"]); iou_nms = float(preset["iou_nms"])
            use_wbf = bool(preset.get("use_wbf", True))
            wbf_alpha = float(self.advanced_params.get("wbf_alpha", DEFAULT_WBF_ALPHA)) if self.advanced_override else DEFAULT_WBF_ALPHA
            wbf_auto_val = auto_wbf_iou(int(self.quality.get()), iou_nms)
            wbf_manual = self.advanced_params.get("wbf_iou", None) if self.advanced_override else None
            # seam params:
            seam_iou_low   = self.advanced_params.get("seam_iou_low", DEFAULT_SEAM_IOU_LOW) if self.advanced_override else DEFAULT_SEAM_IOU_LOW
            seam_band_fact = self.advanced_params.get("seam_band_factor", DEFAULT_SEAM_BAND_FACTOR) if self.advanced_override else DEFAULT_SEAM_BAND_FACTOR
            seam_weight    = self.advanced_params.get("seam_weight", DEFAULT_SEAM_WEIGHT) if self.advanced_override else DEFAULT_SEAM_WEIGHT
            margin_weight  = self.advanced_params.get("margin_weight", DEFAULT_MARGIN_WEIGHT) if self.advanced_override else DEFAULT_MARGIN_WEIGHT

            device = device_auto_str()

            id2name = self.model.names if isinstance(self.model.names, dict) else {i:nm for i,nm in enumerate(self.model.names)}
            selected_idx = self.selected_class_indices()
            select_names = [id2name[i] for i in selected_idx]

            self._log(f"Parametrar: {'ADV' if self.advanced_override else int(self.quality.get())} "
                      f"tile={tile}, overlap={overlap}, conf={conf}, iou_nms={iou_nms}, "
                      f"WBF={use_wbf}, wbf_alpha={wbf_alpha}, wbf_iou={'auto('+str(wbf_auto_val)+')' if wbf_manual is None else wbf_manual}, "
                      f"seam_iou_low={seam_iou_low}, zonfaktor={seam_band_fact}, vikt_marg={margin_weight}, vikt_skarv={seam_weight}, device={device}")
            self._log(f"Valda klasser: {', '.join(select_names)}")
            self._log(f"Visning: {self.overlay_mode.get()}")

            rows, summary_rows = [], []
            n_images = len(images)
            start_time = time.time()

            for idx, img_path in enumerate(images):
                if self.abort_event.is_set(): break
                img = cv2.imread(str(img_path))
                if img is None:
                    self._log(f"[VARN] Kan inte läsa: {img_path.name}")
                    continue
                H, W = img.shape[:2]

                aoi_pts = None
                if aoi_enabled:
                    aoi_pts = load_aoi_points_if_exist(aoi_base_folder, img_path)
                    if aoi_pts is None and n_images == 1:
                        ed = AOIEditor(self, str(img_path)); self.wait_window(ed)
                        aoi_pts = ed.points if len(ed.points)>=3 else [[0,0],[W,0],[W,H],[0,H]]
                        save_aoi_json_and_mask(aoi_base_folder, img_path, aoi_pts)

                tiles_dir = ensure_dir(tiles_root / img_path.stem)
                raw_dir = ensure_dir(raw_root / img_path.stem)

                def prog(tile_idx, total_tiles):
                    frac_img = tile_idx / max(1,total_tiles)
                    frac_all = (idx + frac_img) / n_images
                    elapsed = time.time() - start_time
                    eta = self._eta(elapsed, frac_all)
                    self._set_progress(frac_all*100.0, f"Bild {idx+1}/{n_images} — ruta {tile_idx}/{total_tiles} — ETA {eta}")

                boxes_all, scores_all, cids_all, pre_nms_count, margins_out = detect_image_tiled(
                    self.model, img, selected_idx, conf, tile, overlap, iou_nms, device,
                    tiles_dir=tiles_dir, raw_dir=raw_dir, names_map=id2name,
                    use_wbf=use_wbf, wbf_alpha=wbf_alpha,
                    wbf_iou_manual=wbf_manual, wbf_iou_auto_val=wbf_auto_val,
                    progress_cb=prog, abort_event=self.abort_event,
                    seam_iou_low=seam_iou_low, seam_band_factor=seam_band_fact,
                    seam_weight=seam_weight, margin_weight=margin_weight
                )
                if self.abort_event.is_set(): break

                conf_infer = max(conf - 0.05, 0.05)
                keep_base  = [s >= conf for s in scores_all]
                keep_lower = [s >= conf_infer for s in scores_all]

                def in_aoi(b):
                    if not (aoi_enabled and aoi_pts): return True
                    cx, cy = bbox_center(b); return point_in_polygon(cx, cy, aoi_pts)

                b_base, s_base, c_base = [], [], []
                for b, s, c, k in zip(boxes_all, scores_all, cids_all, keep_base):
                    if k and in_aoi(b): b_base.append(b); s_base.append(s); c_base.append(c)

                b_low = [b for b, s, k in zip(boxes_all, scores_all, keep_lower) if k and in_aoi(b)]
                stability_delta = max(0, len(b_low) - len(b_base))
                stability_ratio = (stability_delta / max(1, len(b_base))) if len(b_base) > 0 else 1.0

                dup_rate = (pre_nms_count / max(1, len(boxes_all)))

                def near_border(b):
                    cx, cy = bbox_center(b)
                    return (cx <= BORDER_PX) or (cy <= BORDER_PX) or (W - cx <= BORDER_PX) or (H - cy <= BORDER_PX)
                border_hits = sum(1 for b in b_base if near_border(b))
                border_risk = border_hits / max(1, len(b_base))

                if s_base:
                    mean_conf = float(np.mean(s_base)); med_conf = float(np.median(s_base))
                    gt07 = sum(1 for s in s_base if s >= 0.70) / len(s_base)
                else:
                    mean_conf = 0.0; med_conf = 0.0; gt07 = 0.0

                qi = 100.0 * (0.5*mean_conf + 0.3*gt07 + 0.2*(1.0 - min(1.0, stability_ratio)))
                qi = float(max(0.0, min(100.0, qi)))

                class_counts = {}
                for cid in c_base:
                    nm = id2name[cid]; class_counts[nm] = class_counts.get(nm, 0) + 1
                total = sum(class_counts.values())

                # ===== JSON: detections (bbox + centroid) =====
                detections = []
                for (x1,y1,x2,y2), s, cid in zip(b_base, s_base, c_base):
                    cx, cy = bbox_center((x1,y1,x2,y2))
                    detections.append({
                        "bbox":[float(x1), float(y1), float(x2), float(y2)],
                        "cx": float(cx), "cy": float(cy),
                        "score": float(s),
                        "class_id": int(cid),
                        "class_name": id2name[cid]
                    })

                ann_path = save_json_collision({
                    "image": str(img_path),
                    "aoi_used": bool(aoi_enabled and aoi_pts),
                    "aoi_points": aoi_pts if (aoi_enabled and aoi_pts) else None,
                    "counts": {**class_counts, "TotalSelected": total},
                    "detections": detections,
                    "metrics": {
                        "mean_conf": mean_conf, "median_conf": med_conf, "p_conf_gt_0_70": gt07,
                        "stability_delta": int(stability_delta),
                        "duplicate_rate": float(dup_rate),
                        "border_risk": float(border_risk),
                        "quality_index": qi
                    },
                    "params": {
                        "conf_base": conf, "conf_infer": conf_infer, "tile": tile, "overlap": overlap, "iou_nms": iou_nms,
                        "use_wbf": use_wbf, "wbf_alpha": wbf_alpha,
                        "wbf_iou": (wbf_manual if wbf_manual is not None else f"auto({wbf_auto_val})"),
                        "seam_iou_low": seam_iou_low, "seam_band_factor": seam_band_fact,
                        "seam_weight": seam_weight, "margin_weight": margin_weight,
                        "overlay_mode": self.overlay_mode.get()
                    }
                }, ensure_dir(ann_dir) / f"{img_path.stem}.json")

                # ===== PREVIEW =====
                preview = img.copy()
                if aoi_enabled and aoi_pts:
                    poly = np.array(aoi_pts, dtype=np.int32); cv2.polylines(preview, [poly], True, (0,255,255), 2)

                mode = self.overlay_mode.get()
                if b_base:
                    if mode == "centroid":
                        # kropki w centroidach (delikatnie)
                        for (x1,y1,x2,y2), cid in zip(b_base, c_base):
                            cx, cy = bbox_center((x1,y1,x2,y2))
                            cv2.circle(preview, (int(cx), int(cy)), 5, (0,255,0), -1, lineType=cv2.LINE_AA)
                    elif mode in ("boxes", "boxes_conf"):
                        show_conf = (mode == "boxes_conf")
                        if sv is not None:
                            det = sv.Detections(xyxy=np.array(b_base, dtype=np.float32),
                                                confidence=np.array(s_base, dtype=np.float32),
                                                class_id=np.array(c_base, dtype=int))
                            if show_conf:
                                labels = [f"{id2name[c]} {s:.2f}" for s, c in zip(s_base, c_base)]
                            else:
                                labels = [f"{id2name[c]}" for c in c_base]
                            preview = sv.BoxCornerAnnotator(thickness=2).annotate(preview, det)
                            preview = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=1).annotate(preview, det, labels=labels)
                        else:
                            for (x1,y1,x2,y2), s, c in zip(b_base, s_base, c_base):
                                cv2.rectangle(preview, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                                label = f"{id2name[c]} {s:.2f}" if show_conf else f"{id2name[c]}"
                                cv2.putText(preview, label, (int(x1), max(14, int(y1)-6)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

                # Overlay z podsumowaniem (większa czcionka)
                overlay = preview.copy()
                lines = []
                for nm in select_names:
                    cnt = class_counts.get(nm, 0)
                    if cnt > 0: lines.append(f"{nm}: {cnt}")
                lines.append(f"Total: {total}")
                if lines:
                    pad = 12; lh = 24
                    w = max([cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0] for t in lines]) + 2*pad
                    h = lh*len(lines) + 2*pad
                    cv2.rectangle(overlay, (10,10), (10+w,10+h), (0,0,0), -1)
                    preview = cv2.addWeighted(overlay, 0.45, preview, 0.55, 0)
                    y = 10 + pad + 18
                    for t in lines:
                        cv2.putText(preview, t, (10+pad, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
                        y += lh

                # zapis PREVIEW z kolizjami
                prv_path = save_image_collision(preview, ensure_dir(prv_dir) / f"{img_path.stem}_annotated.jpg")
                self._log(f"Sparad förhandsvisning: {prv_path.name}")
                if str(ann_path) != str((ensure_dir(ann_dir) / f"{img_path.stem}.json")):
                    self._log(f"Annotation sparad som: {ann_path.name}")

                row = {"image": str(img_path)}
                for nm in select_names:
                    row[nm] = int(class_counts.get(nm, 0))
                row["TotalSelected"] = int(total); rows.append(row)

                summary_rows.append({
                    "image": str(img_path), "TotalSelected": total,
                    "mean_conf": round(mean_conf, 4), "median_conf": round(med_conf, 4),
                    "p_conf_gt_0_70": round(gt07, 4), "stability_delta": int(stability_delta),
                    "duplicate_rate": round(dup_rate, 4), "border_risk": round(border_risk, 4),
                    "quality_index": round(qi, 2),
                })

                self._log(f"OK: {img_path.name} — {total} objekt")

            # CSV/summary/meta — zapisy kolizyjne
            counts_path = save_csv_collision(pd.DataFrame(rows), outp / "counts.csv")
            summary_path = save_csv_collision(pd.DataFrame(summary_rows), outp / "summary.csv")
            meta_path = save_json_collision({
                "started_at": t0, "finished_at": time.time(),
                "params": {"quality": int(self.quality.get()) if not self.advanced_override else "ADV",
                           **preset, "device_auto": device,
                           "wbf_alpha": wbf_alpha,
                           "wbf_iou": (wbf_manual if wbf_manual is not None else f"auto({wbf_auto_val})"),
                           "seam_iou_low": seam_iou_low, "seam_band_factor": seam_band_fact,
                           "seam_weight": seam_weight, "margin_weight": margin_weight,
                           "overlay_mode": self.overlay_mode.get()},
                "selected_classes": select_names,
                "input": {"selected_files": [str(p) for p in images] if self.selected_files else None,
                          "folder": str(aoi_base_folder)},
                "output_dir": str(outp), "aoi_enabled": aoi_enabled,
                "temps": ["temp/extracted_models", "temp/tiles", "temp/raw_preds"]
            }, outp / "run_metadata.json")

            self._log(f"Counts CSV: {counts_path.name}")
            self._log(f"Summary CSV: {summary_path.name}")
            self._log(f"Metadata: {meta_path.name}")

            if self.abort_event.is_set():
                self._set_progress(None, "Avbrutet.")
                self._log("=== AVBRUTET av användaren ===")
            else:
                self._set_progress(100.0, "Klart.")
                self._log(f"Klart. Resultat: {outp}")
        except Exception as e:
            self._log(f"[FEL] {e}")
        finally:
            # sygnał: wątek się zakończył (dla resetu po ABORT)
            self.worker_done.set()
            self.btn_start.config(state="normal")
            self.btn_abort.config(state="disabled")

    # ---------- helpers ----------
    def _log(self, msg: str):
        try:
            self.log.insert("end", msg + "\n"); self.log.see("end")
        except Exception:
            pass

    def _set_progress(self, percent: float|None, text: str):
        def _upd():
            if percent is not None:
                self.progress_var.set(max(0.0, min(100.0, percent)))
            self.progress_label.set(text)
        try:
            self.after(0, _upd)
        except Exception:
            pass

    def _eta(self, elapsed_s: float, progress_frac: float) -> str:
        if progress_frac <= 1e-6: return "--:--"
        total = elapsed_s / progress_frac
        remain = max(0.0, total - elapsed_s)
        m = int(remain // 60); s = int(remain % 60)
        return f"{m:02d}:{s:02d}"

# ===================== MAIN =====================
if __name__ == "__main__":
    app = App(); app.mainloop()
