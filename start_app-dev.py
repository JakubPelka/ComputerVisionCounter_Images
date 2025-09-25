# start_app.py — launcher (modular), pewny pasek postępu, poprawione AOI i lista klas
from __future__ import annotations

import os, sys, json, time, errno
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# ---- prefer local /pkgs first ----
def _add_local_pkgs():
    here = Path(__file__).parent.resolve()
    for d in ("pkgs", "_pkgs"):
        p = here / d
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
_add_local_pkgs()

# ---- optional deps for legacy PT path ----
try:
    import cv2, numpy as np
    from ultralytics import YOLO
except Exception:
    cv2 = None; np = None; YOLO = None

# ---- project modules ----
from app_core import (
    InferConfig, ModelEngine, collect_images,
    save_csv, save_json, save_run_metadata
)
from widgets import ScrollableFrame, AOIEditor
import ui_panels

APP_TITLE = "ComputerVision Counter — Count anything without coding"

# ===== Presets & constants =====
QUALITY_PRESETS = {
    1: {"tile": 640,  "overlap": 0.15, "conf": 0.40, "iou_nms": 0.65, "use_wbf": True},
    2: {"tile": 896,  "overlap": 0.25, "conf": 0.45, "iou_nms": 0.60, "use_wbf": True},
    3: {"tile": 1024, "overlap": 0.30, "conf": 0.50, "iou_nms": 0.55, "use_wbf": True},
    4: {"tile": 1280, "overlap": 0.45, "conf": 0.60, "iou_nms": 0.50, "use_wbf": True},
    5: {"tile": 2560, "overlap": 0.60, "conf": 0.75, "iou_nms": 0.40, "use_wbf": True},  # ULTRA
}
DEFAULT_QUALITY = 5
DEFAULT_WBF_ALPHA = 0.20
DEFAULT_SEAM_IOU_LOW = 0.30
DEFAULT_SEAM_BAND_FACTOR = 0.10
DEFAULT_SEAM_WEIGHT = 0.35
DEFAULT_MARGIN_WEIGHT = 0.25

def auto_wbf_iou(q, nms):
    return 0.60 if int(q) == 5 else max(0.55, float(nms))

# ===== helpers (legacy PT) =====
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

def _bbox_center(b): return (0.5*(b[0]+b[2]), 0.5*(b[1]+b[3]))

def _select_torch_device(pref: str) -> str:
    # "auto" => "0" jeśli CUDA dostępne inaczej "cpu"
    try:
        import torch
        if pref and pref not in ("auto",):
            return pref
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu" if (pref in ("auto","")) else pref

# ----- AOI helpers -----
def _build_aoi_mask(h, w, aois):
    """Union mask of all polygons in aois (list of {'polygon':[(x,y),...]})"""
    if cv2 is None or np is None or not aois:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    polys = []
    for a in aois:
        poly = a.get("polygon") or []
        if len(poly) >= 3:
            polys.append(np.array(poly, dtype=np.int32))
    if polys:
        cv2.fillPoly(mask, polys, 255)
    return mask

def _persist_aoi_for_images(input_root: Path, images: list[Path], aoi_map: dict, log_fn):
    """Save AOI .json + .png mask per image (union of polygons), ignore EEXIST."""
    if not aoi_map or cv2 is None:
        return
    aoi_dir = input_root / "aoi"; mask_dir = input_root / "aoi_masks"
    try:
        aoi_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if getattr(e, "winerror", None) not in (None, 183) and getattr(e, "errno", None) != errno.EEXIST:
            raise
    for p in images:
        aois = aoi_map.get(str(p), [])
        if not aois: continue
        try:
            with open(aoi_dir / f"{p.stem}.json", "w", encoding="utf-8") as f:
                json.dump({"image": p.name, "aois": aois}, f, ensure_ascii=False, indent=2)
            im = cv2.imread(str(p))
            if im is not None:
                h,w = im.shape[:2]
                mask = _build_aoi_mask(h, w, aois)
                if mask is not None:
                    cv2.imwrite(str(mask_dir / f"{p.stem}.png"), mask)
            log_fn(f"[AOI] Persisted for {p.name} ({sum(len(a.get('polygon',[])) for a in aois)} pts total)")
        except Exception as e:
            log_fn(f"[AOI][WARN] Persist failed for {p.name}: {e}")

# ===== App =====
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1120x880"); self.minsize(980,760)

        self.ScrollableFrame = ScrollableFrame  # injected for ui_panels

        # selections
        self.input_dir = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value="")
        self.weights_path = tk.StringVar(value="")
        self.selected_files: list[Path] = []

        self.engine_var = tk.StringVar(value="auto")  # auto/pt/onnx
        self.device_var = tk.StringVar(value="auto")

        # quality / advanced
        self.quality = tk.IntVar(value=DEFAULT_QUALITY)
        self.advanced_override = False
        self.advanced_params = {
            "tile": QUALITY_PRESETS[DEFAULT_QUALITY]["tile"],
            "overlap": QUALITY_PRESETS[DEFAULT_QUALITY]["overlap"],
            "conf": QUALITY_PRESETS[DEFAULT_QUALITY]["conf"],
            "iou_nms": QUALITY_PRESETS[DEFAULT_QUALITY]["iou_nms"],
            "use_wbf": True,
            "wbf_alpha": DEFAULT_WBF_ALPHA,
            "wbf_iou": None, "wbf_auto": True,
            "seam_iou_low": DEFAULT_SEAM_IOU_LOW,
            "seam_band_factor": DEFAULT_SEAM_BAND_FACTOR,
            "seam_weight": DEFAULT_SEAM_WEIGHT,
            "margin_weight": DEFAULT_MARGIN_WEIGHT,
        }

        # AOI
        self.use_aoi = tk.BooleanVar(value=False)
        self.require_aoi_all = tk.BooleanVar(value=False)
        self.persist_aoi = tk.BooleanVar(value=True)
        self.aoi_mode = tk.StringVar(value="centroid")       # 'centroid' | 'box'
        self.aoi_box_frac = tk.DoubleVar(value=0.20)
        self.aoi_map: dict[str, list[dict]] = {}

        # visualization
        self.overlay_mode = tk.StringVar(value="boxes_conf")  # 'boxes'|'boxes_conf'|'centroid'
        self.annotate = tk.BooleanVar(value=True)
        self.draw_centroid = tk.BooleanVar(value=False)

        # classes
        self.show_class_checkboxes = tk.BooleanVar(value=True)
        self.require_class_selection = tk.BooleanVar(value=False)
        self.class_names: dict[int,str] = {}
        self.class_vars: dict[int, tk.BooleanVar] = {}
        self._class_cb_widgets = []

        # progress & logging
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label = tk.StringVar(value="Ready.")
        self._stop = False
        self._logfile: Path | None = None

        # build UI
        ui_panels.build_main_ui(self)

        # hotkeys
        self.bind("<Control-Return>", lambda e: self.start())
        self.bind("<Control-a>", lambda e: self._open_aoi_editor())

        self._log("Ready.")

    # ---------- pickers ----------
    def browse_input(self):
        d = filedialog.askdirectory(title="Select input folder with images")
        if not d: return
        self.input_dir.set(d)
        imgs = collect_images(Path(d))
        self.progress_label.set(f"{len(imgs)} images ready")
        self._refresh_files_label()

    def browse_files(self):
        files = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")]
        )
        if not files: return
        self.selected_files = [Path(f) for f in files]
        self.progress_label.set(f"{len(self.selected_files)} images selected")
        self._refresh_files_label()

    def clear_files(self):
        self.selected_files = []
        self._refresh_files_label()

    def _refresh_files_label(self):
        try:
            if self.selected_files:
                self.files_label.config(text=f"{len(self.selected_files)} files selected")
            else:
                root = Path(self.input_dir.get().strip()) if self.input_dir.get().strip() else None
                if root and root.exists():
                    self.files_label.config(text=f"{len(collect_images(root))} images in folder")
                else:
                    self.files_label.config(text="— no files selected —")
        except Exception: pass

    def browse_output(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d: self.output_dir.set(d)

    def browse_weights(self):
        p = filedialog.askopenfilename(
            title="Select weights (.pt or .onnx)",
            filetypes=[("Models","*.pt *.onnx"), ("All files","*.*")]
        )
        if not p: return
        self.weights_path.set(p)
        self._log(f"Model: {Path(p).name}")
        try:
            cfg = InferConfig(model_path=p, engine=self.engine_var.get(), device=self.device_var.get())
            eng = ModelEngine(cfg)
            self.class_names = eng.available_classes()
            self._build_class_checkboxes()
            self._log(f"Loaded {len(self.class_names)} classes from engine.")
        except Exception as e:
            self._log(f"[WARN] Could not read classes from engine: {e}")

    # ---------- AOI ----------
    def _on_toggle_use_aoi(self):
        if self.use_aoi.get():
            self._open_aoi_editor()

    def _open_aoi_editor(self):
        imgs = self._resolve_inputs()
        if not imgs:
            messagebox.showinfo("AOI", "Select input images first."); return
        idx = 0
        top = tk.Toplevel(self); top.title("AOI — define polygons per image"); top.geometry("1100x840")
        nav = tk.Frame(top); nav.pack(fill="x", pady=4)
        idx_var = tk.StringVar(value=f"1/{len(imgs)}")
        tk.Button(nav, text="⟵ Prev", command=lambda: load_idx(idx-1)).pack(side="left")
        tk.Button(nav, text="Next ⟶", command=lambda: load_idx(idx+1)).pack(side="left", padx=6)
        tk.Label(nav, textvariable=idx_var).pack(side="left", padx=10)
        tk.Button(nav, text="Save AOI for current", command=lambda: save_current()).pack(side="left", padx=10)

        holder = tk.Frame(top); holder.pack(fill="both", expand=True)
        editor = AOIEditor(holder); editor.pack(fill="both", expand=True)

        def load_idx(i):
            nonlocal idx
            i = max(0, min(i, len(imgs)-1)); idx = i
            p = imgs[i]
            editor.load_image(str(p))
            aois = self.aoi_map.get(str(p), [])
            if aois: editor.set_aois(aois)
            idx_var.set(f"{i+1}/{len(imgs)}")
        def save_current():
            p = imgs[idx]; self.aoi_map[str(p)] = editor.get_aois()
            self._log(f"[AOI] Saved {len(self.aoi_map[str(p)])} polygons for {Path(p).name}")
        load_idx(0)

    # ---------- classes ----------
    def _build_class_checkboxes(self):
        for w in self._class_cb_widgets:
            try: w.destroy()
            except Exception: pass
        self._class_cb_widgets = []
        self.class_vars = {}
        names = [self.class_names[i] for i in sorted(self.class_names.keys())] if self.class_names else []
        for i, nm in enumerate(names):
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(self.classes_container, text=nm, variable=var, anchor="w", padx=6)
            self.class_vars[i] = var; self._class_cb_widgets.append(cb)
            cb.grid(row=0, column=i, sticky="w", padx=6, pady=4)  # wstępnie, zaraz przepakujemy
        self._reflow_class_grid()

    def _reflow_class_grid(self, width: int=None):
        if not self._class_cb_widgets: return
        try: cw = int(width or self.classes_container.winfo_width() or 800)
        except Exception: cw = 800
        cols = max(2, min(8, cw // 220))
        for idx, w in enumerate(self._class_cb_widgets):
            r,c = divmod(idx, cols)
            w.grid_configure(row=r, column=c, sticky="w", padx=6, pady=4)
        for c in range(cols):
            try: self.classes_container.grid_columnconfigure(c, weight=1)
            except Exception: pass

    def _selected_classes(self):
        if not self.show_class_checkboxes.get() or not self.class_vars: return None
        sel = [i for i,v in self.class_vars.items() if v.get()]
        return sel if sel else None

    # ---------- logging ----------
    def _set_logfile(self, outdir: Path):
        try:
            outdir.mkdir(parents=True, exist_ok=True)
            self._logfile = outdir / "run.log"
            with open(self._logfile, "a", encoding="utf-8") as f:
                f.write("\n==== New run ====\n")
        except Exception:
            self._logfile = None

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            self.log.insert("end", line + "\n"); self.log.see("end")
        except Exception: pass
        try:
            if self._logfile:
                with open(self._logfile, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception: pass

    # ---------- inputs resolve ----------
    def _resolve_inputs(self) -> list[Path]:
        if self.selected_files:
            return [Path(p) for p in self.selected_files]
        inp = Path(self.input_dir.get().strip()) if self.input_dir.get().strip() else None
        if not inp or not inp.exists(): return []
        return collect_images(inp)

    # ---------- run ----------
    def start(self):
        try:
            imgs = self._resolve_inputs()
            if not imgs:
                messagebox.showerror("Input", "Select a valid image folder or files."); return
            model = self.weights_path.get().strip()
            if not model:
                messagebox.showerror("Model", "Select a valid weights file (.pt or .onnx)."); return
            if self.require_class_selection.get() and not self._selected_classes():
                messagebox.showwarning("Classes", "Select at least one class or disable the requirement."); return
            if self.use_aoi.get() and self.require_aoi_all.get():
                missing = [p for p in imgs if str(p) not in self.aoi_map or not self.aoi_map[str(p)]]
                if missing:
                    messagebox.showwarning("AOI", f"AOI missing for {len(missing)} images."); return

            outdir = Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else (
                (imgs[0].parent if self.selected_files else Path(self.input_dir.get().strip())) / "results"
            )
            self._set_logfile(outdir); self._log(f"Output → {outdir}")
            self._log(f"Using model: {model}  | engine={self.engine_var.get()} device={self.device_var.get()}")

            # base params
            base = QUALITY_PRESETS.get(int(self.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY]).copy()
            if self.advanced_override and self.advanced_params:
                base.update(self.advanced_params)
                if base.get("wbf_auto", True):
                    base["wbf_iou"] = auto_wbf_iou(int(self.quality.get()), base["iou_nms"])

            # Persist AOIs to input (optional)
            if self.use_aoi.get() and self.aoi_map and self.persist_aoi.get() and cv2 is not None:
                root = imgs[0].parent if self.selected_files else Path(self.input_dir.get().strip())
                _persist_aoi_for_images(root, imgs, self.aoi_map, self._log)

            # choose path
            use_legacy_pt = (self.engine_var.get() in ("auto","pt") and model.lower().endswith(".pt") and YOLO is not None)
            self._log(f"Path: {'legacy-PT' if use_legacy_pt else 'engine-core'}")

            self.btn_start.config(state="disabled"); self.btn_abort.config(state="normal")
            self._stop = False; self.progress_var.set(0.0); self.progress_label.set("Running…")
            self.update_idletasks()

            if use_legacy_pt:
                totals = self._run_legacy_pt(imgs, outdir, model, base)
            else:
                totals = self._run_engine_core(imgs, outdir, model, base)

            self.progress_label.set(f"Done. Output: {outdir}")
            self._log(f"Done. Totals: {totals}")
            messagebox.showinfo("Done", f"Processed {len(imgs)} images.\nSaved to: {outdir}")
        except Exception as e:
            messagebox.showerror("Error", str(e)); self._log(f"[ERROR] {e}")
        finally:
            self.btn_start.config(state="normal"); self.btn_abort.config(state="disabled")

    def abort(self): self._stop = True; self._log("=== ABORT requested ===")

    def _on_progress(self, i, n, eta_sec):
        # wywoływane z app_core.predict_batch po każdym obrazie (patrz progress_cb) :contentReference[oaicite:3]{index=3}
        pct = 100.0 * (i / max(1,n)); self.progress_var.set(pct)
        m = int(eta_sec // 60); s = int(eta_sec % 60)
        self.progress_label.set(f"Image {i}/{n} — ETA {m:02d}:{s:02d}")
        self.update_idletasks()  # wymuś repaint

    # ---------- engine-core path ----------
    def _run_engine_core(self, imgs, outdir: Path, model, base):
        cfg = InferConfig(
            model_path=model, engine=self.engine_var.get(), device=self.device_var.get(),
            conf=float(base["conf"]), iou=float(base["iou_nms"]), imgsz=int(base["tile"]),
            classes=self._selected_classes(),
            aoi_mode=self.aoi_mode.get(), aoi_box_frac=float(self.aoi_box_frac.get()),
            annotate=bool(self.annotate.get()), draw_centroid=bool(self.draw_centroid.get()),
            use_tiling=True, tile=int(base["tile"]), overlap=float(base["overlap"]),
            use_wbf=bool(base.get("use_wbf", True)),
            wbf_iou=float(base.get("wbf_iou", auto_wbf_iou(int(self.quality.get()), base["iou_nms"]))),
            wbf_alpha=float(base.get("wbf_alpha", DEFAULT_WBF_ALPHA)),
            seam_band_factor=float(base.get("seam_band_factor", DEFAULT_SEAM_BAND_FACTOR)),
            seam_weight=float(base.get("seam_weight", DEFAULT_SEAM_WEIGHT)),
            overlay_mode=self.overlay_mode.get(), persist_aoi_to_input=bool(self.persist_aoi.get()),
        )
        engine = ModelEngine(cfg)
        self._log(f"[engine] names={engine.available_classes()}")
        per_image, totals = engine.predict_batch(
            imgs, aoi_map=self.aoi_map if self.use_aoi.get() else {},
            outdir=outdir, annotate=cfg.annotate,
            progress_cb=self._on_progress, abort_cb=lambda: self._stop
        )
        # Save CSV/JSON
        class_names = sorted({k for _, cnt in per_image for k in cnt.keys()})
        rows = [[path] + [cnt.get(c,0) for c in class_names] for path, cnt in per_image]
        save_csv(rows, header=["image_path"]+class_names, out_path=outdir/"results_per_image.csv")
        save_json(totals, out_path=outdir/"results_totals.json")
        save_run_metadata(outdir, imgs, cfg, totals)
        return totals

    # ---------- legacy-PT path (Ultralytics + tiling, multi-AOI) ----------
    def _run_legacy_pt(self, imgs, outdir: Path, model, base):
        if cv2 is None or np is None or YOLO is None:
            raise RuntimeError("Legacy PT path requires opencv-python, numpy, ultralytics installed.")
        outdir.mkdir(parents=True, exist_ok=True)
        ann_dir = (outdir / "annotations"); ann_dir.mkdir(exist_ok=True, parents=True)
        prv_dir = (outdir / "annotated"); prv_dir.mkdir(exist_ok=True, parents=True)

        device = _select_torch_device(self.device_var.get())
        self._log(f"[legacy-pt] device={device}")
        model_pt = YOLO(model)

        tile = int(base["tile"]); overlap=float(base["overlap"])
        conf=float(base["conf"]); iou=float(base["iou_nms"])
        classes = self._selected_classes()
        show_conf = (self.overlay_mode.get()=="boxes_conf")
        start = time.time()

        totals = {}
        per_rows = []  # for CSV
        for idx, p in enumerate(imgs, 1):
            if self._stop: break
            img = cv2.imread(str(p))
            if img is None: self._log(f"[WARN] cannot read {p}"); continue
            H,W = img.shape[:2]

            # ---- tiling inference ----
            step = max(1, int(tile*(1.0-overlap)))
            xs = list(range(0, W, step)); ys = list(range(0, H, step))
            all_boxes=[]; all_scores=[]; all_cids=[]
            for r_i, yy in enumerate(ys, 1):
                for xx in xs:
                    if self._stop: break
                    roi = img[yy:min(yy+tile,H), xx:min(xx+tile,W)]
                    res = model_pt.predict(source=roi, conf=max(conf-0.05,0.05),
                                           imgsz=tile, device=device, classes=classes, verbose=False)
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
                # progress (per row)
                frac_img = r_i / max(1,len(ys))
                frac_all = ((idx-1)+frac_img)/len(imgs)
                elapsed = time.time()-start
                tot = max(1e-9, elapsed / max(1e-9, frac_all))
                remain = max(0.0, tot - elapsed)
                m = int(remain // 60); s = int(remain % 60)
                self.progress_var.set(frac_all*100.0)
                self.progress_label.set(f"Image {idx}/{len(imgs)} — ETA {m:02d}:{s:02d}")
                self.update_idletasks()

            # NMS
            if all_boxes:
                keep = _nms_numpy(np.array(all_boxes,np.float32), np.array(all_scores,np.float32), iou_thr=iou)
                all_boxes = [all_boxes[k] for k in keep]
                all_scores = [all_scores[k] for k in keep]
                all_cids = [all_cids[k] for k in keep]

            # AOI filtering (union)
            aois = (self.aoi_map.get(str(p)) or []) if self.use_aoi.get() else []
            if aois:
                mask = _build_aoi_mask(H, W, aois)
                kept_idx = []
                if self.aoi_mode.get() == "box" and mask is not None:
                    thr = float(self.aoi_box_frac.get() or 0.0)
                    for i, bb in enumerate(all_boxes):
                        x1,y1,x2,y2 = [max(0,int(v)) for v in bb]
                        area = max(1, (x2-x1)*(y2-y1))
                        inter = int((mask[y1:y2, x1:x2] > 0).sum())
                        if inter / float(area) >= thr:
                            kept_idx.append(i)
                else:  # centroid inside
                    if mask is not None:
                        for i, bb in enumerate(all_boxes):
                            cx,cy = map(int, _bbox_center(bb))
                            if 0 <= cx < W and 0 <= cy < H and mask[cy, cx] > 0:
                                kept_idx.append(i)
                all_boxes = [all_boxes[i] for i in kept_idx]
                all_scores= [all_scores[i] for i in kept_idx]
                all_cids  = [all_cids[i]   for i in kept_idx]

            # counts & outputs
            id2name = {i:str(i) for i in sorted(set(all_cids))}
            counts = {}
            for c in all_cids:
                nm = id2name.get(c, str(c)); counts[nm] = counts.get(nm,0)+1
            for nm, v in counts.items():
                totals[nm] = totals.get(nm,0)+v

            # per-image JSON (legacy path)
            dets = []
            for bb,sc,cc in zip(all_boxes, all_scores, all_cids):
                cx,cy = _bbox_center(bb)
                dets.append({"bbox":bb, "score":sc, "class_id":int(cc),
                             "class_name": id2name.get(cc,str(cc)), "cx":cx, "cy":cy})
            with open(ann_dir/f"{p.stem}.json","w",encoding="utf-8") as f:
                json.dump({"image": str(p), "counts": counts, "detections": dets}, f, ensure_ascii=False, indent=2)

            # preview
            preview = img.copy()
            if aois:
                polys = [np.array(a.get("polygon", []), np.int32) for a in aois if len(a.get("polygon",[]))>=3]
                for poly in polys:
                    cv2.polylines(preview, [poly], True, (0,255,255), 2)
            for bb,sc,cc in zip(all_boxes, all_scores, all_cids):
                x1,y1,x2,y2 = map(int, bb)
                cv2.rectangle(preview,(x1,y1),(x2,y2),(0,255,0),2)
                if show_conf:
                    nm = id2name.get(cc, str(cc))
                    cv2.putText(preview, f"{nm} {sc:.2f}", (x1,max(14,y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            if counts:
                txt = "  ".join([f"{k}:{v}" for k,v in counts.items()])
                (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                x2,y2 = W-10, H-10
                cv2.rectangle(preview, (x2-tw-12,y2-th-10),(x2,y2),(0,0,0),-1)
                cv2.putText(preview, txt, (x2-tw-8,y2-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2, cv2.LINE_AA)
            cv2.imwrite(str(prv_dir/f"{p.stem}_annotated.jpg"), preview)

            # CSV row (legacy path)
            all_names = sorted(set(id2name.values()))
            row = [str(p)] + [counts.get(nm,0) for nm in all_names]
            header = ["image_path"] + all_names
            per_rows.append((header, row))

        # save totals + CSV (per image)
        save_json(totals, out_path=outdir/"results_totals.json")
        if per_rows:
            # merge header across rows
            final_names = sorted({n for (hdr,_r) in per_rows for n in hdr[1:]})
            rows = []
            for (_hdr, r) in per_rows:
                path = r[0]; local = dict(zip(_hdr[1:], r[1:]))
                rows.append([path] + [local.get(n,0) for n in final_names])
            save_csv(rows, header=["image_path"]+final_names, out_path=outdir/"results_per_image.csv")
        return totals

    # ---------- Advanced ----------
    def _update_preset_label(self):
        try:
            p = QUALITY_PRESETS.get(int(self.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY])
            self.preset_label.config(text=f"tile={p['tile']}  overlap={p['overlap']}  conf={p['conf']}  nms={p['iou_nms']}  WBF={p['use_wbf']}")
        except Exception:
            pass

    def open_advanced(self):
        import tkinter as tk
        win = tk.Toplevel(self); win.title("Advanced options"); win.geometry("660x600")

        tk.Label(win, text="Current preset (from the slider):").pack(anchor="w", padx=8, pady=(8,2))
        p = QUALITY_PRESETS.get(int(self.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY])
        preset_txt = tk.StringVar(value=f"tile={p['tile']}  overlap={p['overlap']}  conf={p['conf']}  nms={p['iou_nms']}  WBF={p['use_wbf']}")
        tk.Entry(win, textvariable=preset_txt, state="readonly").pack(fill="x", padx=8)

        current_auto_wbf = auto_wbf_iou(int(self.quality.get()), float(p["iou_nms"]))

        tk.Label(win, text="Override (leave blank = use preset/auto):").pack(anchor="w", padx=8, pady=(10,2))
        def row(lbl, var):
            f = tk.Frame(win); f.pack(fill="x", pady=3)
            tk.Label(f, text=lbl, width=30, anchor="w").pack(side="left")
            tk.Entry(f, textvariable=var, width=18).pack(side="left"); return f

        base = self.advanced_params if self.advanced_override else {
            **p, "wbf_alpha": DEFAULT_WBF_ALPHA, "wbf_iou": None, "wbf_auto": True,
            "seam_iou_low": DEFAULT_SEAM_IOU_LOW, "seam_band_factor": DEFAULT_SEAM_BAND_FACTOR,
            "seam_weight": DEFAULT_SEAM_WEIGHT, "margin_weight": DEFAULT_MARGIN_WEIGHT
        }
        S = lambda k, d="": tk.StringVar(value=str(base.get(k, d)))
        var_tile, var_ov, var_conf, var_nms = S("tile"), S("overlap"), S("conf"), S("iou_nms")
        var_wbf = tk.BooleanVar(value=bool(base.get("use_wbf", True)))
        var_alpha = S("wbf_alpha", DEFAULT_WBF_ALPHA)
        var_wbf_iou = tk.StringVar(value="" if base.get("wbf_iou", None) is None else str(base.get("wbf_iou")))
        var_wbf_auto = tk.BooleanVar(value=bool(base.get("wbf_auto", True)))
        var_seam_iou_low = S("seam_iou_low", DEFAULT_SEAM_IOU_LOW)
        var_seam_band = S("seam_band_factor", DEFAULT_SEAM_BAND_FACTOR)
        var_seam_w = S("seam_weight", DEFAULT_SEAM_WEIGHT)
        var_margin_w = S("margin_weight", DEFAULT_MARGIN_WEIGHT)

        row("Tile size", var_tile); row("Overlap (0..0.9)", var_ov); row("Confidence", var_conf); row("NMS IoU", var_nms)
        f = tk.Frame(win); f.pack(fill="x", pady=3)
        tk.Checkbutton(f, text="Use WBF", variable=var_wbf).pack(side="left", padx=(0,10))
        tk.Label(f, text="WBF alpha:").pack(side="left"); tk.Entry(f, textvariable=var_alpha, width=8).pack(side="left")
        tk.Checkbutton(f, text="Auto WBF IoU", variable=var_wbf_auto).pack(side="left", padx=(12,10))
        tk.Label(f, text="WBF IoU (manual):").pack(side="left"); tk.Entry(f, textvariable=var_wbf_iou, width=8).pack(side="left")

        row("Seam IoU low", var_seam_iou_low); row("Seam band factor", var_seam_band)
        row("Seam weight", var_seam_w); row("Margin weight", var_margin_w)

        btns = tk.Frame(win); btns.pack(fill="x", pady=10)

        def get_or(sv: tk.StringVar, cast, name):
            txt = sv.get().strip()
            return cast(txt) if txt != "" else base.get(name)

        def apply_override():
            try:
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
                    "seam_band_factor": float(var_seam_band.get()) if var_seam_band.get().strip()!="" else DEFAULT_SEAM_BAND_FACTOR,
                    "seam_weight": float(var_seam_w.get()) if var_seam_w.get().strip()!="" else DEFAULT_SEAM_WEIGHT,
                    "margin_weight": float(var_margin_w.get()) if var_margin_w.get().strip()!="" else DEFAULT_MARGIN_WEIGHT,
                }
                self.advanced_override = True
                self._log(f"[ADV] Override enabled. Auto WBF IoU = {self.advanced_params['wbf_auto']}."); win.destroy()
            except Exception as e:
                messagebox.showerror("Advanced", str(e))
        def reset_to_preset():
            self.advanced_override = False; self._log("[ADV] Preset restored from the slider."); win.destroy()
        def apply_anti_seam():
            cur = QUALITY_PRESETS.get(int(self.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY])
            self.advanced_params = {
                "tile": int(var_tile.get()) if var_tile.get().strip() else int(cur["tile"]),
                "overlap": 0.55, "conf": float(var_conf.get()) if var_conf.get().strip() else float(cur["conf"]),
                "iou_nms": float(var_nms.get()) if var_nms.get().strip() else float(cur["iou_nms"]),
                "use_wbf": True, "wbf_alpha": 0.35, "wbf_iou": 0.60, "wbf_auto": False,
                "seam_iou_low": 0.40, "seam_band_factor": 0.12, "seam_weight": 0.45, "margin_weight": 0.30,
            }
            self.advanced_override = True; self._log("[ADV] Anti-seam preset applied."); win.destroy()

        tk.Button(btns, text="Apply override", command=apply_override).pack(side="left", padx=6)
        tk.Button(btns, text="Restore preset", command=reset_to_preset).pack(side="left", padx=6)
        tk.Button(btns, text="Anti-seam (dedup)", command=apply_anti_seam).pack(side="left", padx=6)


if __name__ == "__main__":
    App().mainloop()
