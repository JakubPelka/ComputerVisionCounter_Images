# start_app.py — main app entry for ComputerVision Counter
from __future__ import annotations
import sys, json, time, threading, traceback, os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

def _add_local_pkgs():
    here = Path(__file__).parent.resolve()
    for d in ("pkgs", "_pkgs"):
        p = here / d
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
_add_local_pkgs()

try:
    import cv2
except Exception:
    cv2 = None

try:
    from geo_export import export_geojson_for_image
except Exception:
    export_geojson_for_image = None

from app_core import (
    InferConfig, ModelEngine, collect_images,
    save_csv, save_json, save_run_metadata
)
from widgets import ScrollableFrame, AOIEditor
import ui_panels
from ui_advanced import open_advanced
## Import both names if available, then provide a safe wrapper used by this file.
from legacy_pt_runner import run_legacy_pt  # always
try:
    from legacy_pt_runner import build_union_mask as _build_union_mask
except Exception:
    _build_union_mask = None
try:
    from legacy_pt_runner import build_union_masks as _build_union_masks
except Exception:
    _build_union_masks = None

def build_union_mask(h: int, w: int, aois):
    """
    ## Tolerant wrapper so start_app.py never breaks on singular/plural export.
    """
    if _build_union_mask is not None:
        return _build_union_mask(h, w, aois)
    if _build_union_masks is not None:
        return _build_union_masks(h, w, aois)
    # Last-resort local implementation (keeps AOI saving working even if imports failed)
    try:
        import numpy as _np, cv2 as _cv2
        m = _np.zeros((h, w), dtype=_np.uint8)
        for a in (aois or []):
            pts = a.get("polygon") or a.get("points") or a.get("pts") or []
            P = _np.asarray(pts, dtype=_np.float32)
            if P.ndim == 2 and P.shape[1] == 2 and len(P) >= 3:
                P[:,0] = _np.clip(P[:,0], 0, w-1)
                P[:,1] = _np.clip(P[:,1], 0, h-1)
                _cv2.fillPoly(m, [P.astype(_np.int32)], 255)
        return m if m.any() else None
    except Exception:
        return None

APP_TITLE = "ComputerVision Counter — Count anything without coding"

# -------- presets / defaults (shared with ui_advanced) --------
BUILTIN_PRESETS = {
    "Fast":     {"tile": 896,  "overlap": 0.25, "conf": 0.45, "iou_nms": 0.60,
                 "use_wbf": True, "wbf_alpha": 0.20, "wbf_iou": None, "wbf_auto": True,
                 "seam_iou_low": 0.30, "seam_band_factor": 0.10, "seam_weight": 0.35, "margin_weight": 0.25},
    "Balanced": {"tile": 1024, "overlap": 0.30, "conf": 0.50, "iou_nms": 0.55,
                 "use_wbf": True, "wbf_alpha": 0.20, "wbf_iou": None, "wbf_auto": True,
                 "seam_iou_low": 0.30, "seam_band_factor": 0.10, "seam_weight": 0.35, "margin_weight": 0.25},
    "Ultra":    {"tile": 2560, "overlap": 0.60, "conf": 0.75, "iou_nms": 0.40,
                 "use_wbf": True, "wbf_alpha": 0.20, "wbf_iou": None, "wbf_auto": True,
                 "seam_iou_low": 0.30, "seam_band_factor": 0.10, "seam_weight": 0.35, "margin_weight": 0.25},
}
DEFAULT_QUALITY_NAME = "Ultra"

def _quality_to_name_snapped(qvalue: float) -> tuple[int, str]:
    v = float(qvalue)
    snap = 1 if v < 1.5 else 2 if v < 2.5 else 3
    name = "Fast" if snap == 1 else "Balanced" if snap == 2 else "Ultra"
    return snap, name

def auto_wbf_iou(qname: str, nms: float) -> float:
    return 0.60 if qname == "Ultra" else max(0.55, float(nms))


class App(tk.Tk):
    CLASS_COL_MIN = 2
    CLASS_COL_MAX = 12
    CLASS_CELL_PX = 160

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("900x700")
        self.minsize(900, 700)

        # selections
        self.input_dir = tk.StringVar(value="")     # no prefill
        self.output_dir = tk.StringVar(value="")    # will default to ./output
        self.weights_path = tk.StringVar(value="")  # no prefill
        self.selected_files: list[Path] = []

        # hidden engine/device controls (auto)
        self.engine_var = tk.StringVar(value="auto")
        self.device_var = tk.StringVar(value="auto")

        # quality / advanced
        self.quality = tk.IntVar(value=3)  # 1..3 – snap to 3 steps
        self.advanced_override = False
        base = BUILTIN_PRESETS[DEFAULT_QUALITY_NAME].copy()
        self.advanced_params = base.copy()
        self.quality.trace_add("write", lambda *_: self._on_quality_changed())

        # AOI
        self.use_aoi = tk.BooleanVar(value=False)
        self.require_aoi_all = tk.BooleanVar(value=False)
        self.aoi_mode = tk.StringVar(value="centroid")  # centroid | box
        self.aoi_box_frac = tk.DoubleVar(value=0.20)
        self.aoi_map: dict[str, list[dict]] = {}

        # viz
        self.overlay_mode = tk.StringVar(value="boxes_conf")
        self.annotate = tk.BooleanVar(value=True)
        self.draw_centroid = tk.BooleanVar(value=False)

        # classes
        self.class_names: dict[int, str] = {}
        self.class_vars: list[tuple[str, tk.BooleanVar, int]] = []
        self._class_cols = 4
        self.classes_scroll: ScrollableFrame | None = None
        self.classes_container = None

        # progress & logging
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label = tk.StringVar(value="Ready.")
        self._smooth_target = 0.0
        self._smooth_job = None

        self._stop = False
        self._logfile: Path | None = None
        self._worker: threading.Thread | None = None

        # ---- only set default OUTPUT to ./output (no prefill for input/weights)
        self._prefill_only_output()

        # build UI
        ui_panels.build_main_ui(self)

        # hotkeys
        self.bind("<Control-Return>", lambda e: self.start())
        self.bind("<Control-a>", lambda e: self._open_aoi_editor())

        self._update_preset_label()
        self._log("Ready.")

    # ---------- only-output prefill ----------
    def _prefill_only_output(self):
        base = Path(__file__).parent.resolve()
        cand_out = base / "output"
        try:
            cand_out.mkdir(exist_ok=True)
        except Exception:
            pass
        self.output_dir.set(str(cand_out))
        # DO NOT touch input_dir / weights_path

    # ---------- file pickers ----------
    def browse_input(self):
        # Start in ./input or current value; do not auto-set beforehand
        start = self.input_dir.get().strip() or str((Path(__file__).parent / "input").resolve())
        d = filedialog.askdirectory(title="Select input folder with images", initialdir=start)
        if not d: return
        self.input_dir.set(d)
        imgs = collect_images(Path(d))
        self.progress_label.set(f"{len(imgs)} images ready")
        self._refresh_files_label()

    def browse_files(self):
        start = self.input_dir.get().strip() or str((Path(__file__).parent / "input").resolve())
        files = filedialog.askopenfilenames(
            title="Select images",
            initialdir=start,
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
        except Exception:
            pass

    def browse_output(self):
        # Start in ./output or current value
        start = self.output_dir.get().strip() or str((Path(__file__).parent / "output").resolve())
        d = filedialog.askdirectory(title="Select output folder", initialdir=start)
        if d: self.output_dir.set(d)

    def browse_weights(self):
        # Start in ./weights (or current weights folder), but do NOT auto-select a file
        cur = self.weights_path.get().strip()
        if cur:
            p = Path(cur)
            start_dir = p.parent if p.exists() else (Path(__file__).parent / "weights")
        else:
            start_dir = Path(__file__).parent / "weights"
        pth = filedialog.askopenfilename(
            title="Select weights (.pt or .onnx)",
            initialdir=str(start_dir.resolve()),
            filetypes=[("Models","*.pt"), ("All files","*.*")]
        )
        if not pth: return
        self.weights_path.set(pth)
        self._log(f"Model: {Path(pth).name}")
        try:
            cfg = InferConfig(model_path=pth, engine=self.engine_var.get(), device=self.device_var.get())
            eng = ModelEngine(cfg)
            self.class_names = eng.available_classes()
            self._populate_classes(self.class_names)
            self._log(f"Loaded {len(self.class_names)} classes from engine.")
        except Exception as e:
            self._log(f"[WARN] Could not read classes from engine: {e}")

    # ---------- presets / AOI ----------
    def _on_quality_changed(self):
        cur = float(self.quality.get())
        snap, name = _quality_to_name_snapped(cur)
        if cur != snap:
            self.quality.set(snap)
            return
        p = BUILTIN_PRESETS[name]
        self.advanced_params.update(p)
        self.advanced_override = False
        self._update_preset_label()
        self._log(f"[ADV] Quality preset from slider: {name} → tile={p['tile']} overlap={p['overlap']} conf={p['conf']} nms={p['iou_nms']}")

    def _on_toggle_use_aoi(self):
        try:
            if self.use_aoi.get():
                self._open_aoi_editor()
        except Exception as e:
            self._log(f"[AOI] toggle error: {e}")

    def _open_aoi_editor(self):
        imgs = self._resolve_inputs()
        if not imgs:
            messagebox.showinfo("AOI", "Select input images first.")
            return

        # Reuse AOIs from disk silently
        self._import_aois_from_folder_for_images(imgs, silent=True)

        idx = 0
        top = tk.Toplevel(self)
        top.title("AOI — define polygons per image")
        top.geometry("1100x840")
        try:
            top.transient(self)
            top.lift()
            top.attributes("-topmost", True)
            top.after(250, lambda: top.attributes("-topmost", False))
        except Exception:
            pass

        def save_current():
            ## Only persist if the editor actually has an image loaded
            try:
                if editor.img_path:
                    p = imgs[idx]
                    self.aoi_map[str(p)] = editor.get_aois()
            except Exception as e:
                self._log(f"[AOI] save_current error: {e}")

        def on_change(_aois):
            save_current()

        nav = tk.Frame(top); nav.pack(fill="x", pady=4)
        idx_var = tk.StringVar(value=f"1/{len(imgs)}")
        tk.Button(nav, text="⟵ Prev", command=lambda: load_idx(idx-1)).pack(side="left")
        tk.Button(nav, text="Next ⟶", command=lambda: load_idx(idx+1)).pack(side="left", padx=6)
        tk.Label(nav, textvariable=idx_var).pack(side="left", padx=10)
        tk.Label(nav, text="(Finish: Ctrl+Enter — Undo vertex: Ctrl+Backspace)", fg="#666").pack(side="left", padx=10)

        holder = tk.Frame(top); holder.pack(fill="both", expand=True)
        editor = AOIEditor(holder, on_change=on_change); editor.pack(fill="both", expand=True)

        def load_idx(i):
            nonlocal idx
            # Only save if we have an actual image to avoid wiping AOIs with an empty list
            if editor.img_path:
                save_current()
            i = max(0, min(i, len(imgs)-1)); idx = i
            p = imgs[i]
            editor.load_image(str(p))
            aois = self.aoi_map.get(str(p), None)
            editor.set_aois(aois if aois is not None else [])
            idx_var.set(f"{i+1}/{len(imgs)}")
            # Force first render in case the window hasn't fully laid out yet
            try:
                top.update_idletasks()
                editor._on_configure()
            except Exception:
                pass

        def on_close():
            ## Always close even if export fails
            try:
                if editor.img_path:
                    save_current()
                self._export_aois_to_input(imgs, force=True)
                if any(self.aoi_map.get(str(p)) for p in imgs):
                    self.use_aoi.set(True)
                    self._log("[AOI] AOIs saved; Use AOI enabled.")
            except Exception as e:
                self._log(f"[AOI] Close/export error: {e}")
            finally:
                try: top.destroy()
                except Exception: pass

        top.protocol("WM_DELETE_WINDOW", on_close)

        # Load the very first image and force a render so the editor never opens blank
        load_idx(0)


    def import_aois_from_input(self):
        imgs = self._resolve_inputs()
        if not imgs:
            messagebox.showinfo("AOI", "Select input images first."); return
        loaded = self._import_aois_from_folder_for_images(imgs, silent=False)
        if loaded > 0:
            self.use_aoi.set(True)
            messagebox.showinfo("AOI", f"Imported AOIs for {loaded} images.")

    def export_aois_now(self):
        imgs = self._resolve_inputs()
        if not imgs:
            messagebox.showinfo("AOI", "Select input images first."); return
        count = self._export_aois_to_input(imgs, force=True)
        messagebox.showinfo("AOI", f"Exported AOIs for {count} images.")

    def _import_aois_from_folder_for_images(self, imgs, silent: bool):
        if not imgs: return 0
        root = imgs[0].parent if self.selected_files else Path(self.input_dir.get().strip())
        folder = root / "aoi"
        if not folder.exists():
            if not silent: self._log(f"[AOI] No 'aoi' folder under {root}")
            return 0
        by_name = {p.name: p for p in imgs}
        loaded = 0
        for jf in sorted(folder.glob("*.json")):
            try:
                data = json.loads(Path(jf).read_text(encoding="utf-8"))
                img_name = data.get("image")
                raw = data.get("aois", None)
                aois = []
                if isinstance(raw, list):
                    for a in raw:
                        pts = a.get("polygon", a.get("points", a.get("pts", [])))
                        if pts and len(pts) >= 3:
                            aois.append({"name": a.get("name","AOI"),
                                         "polygon": [[float(x),float(y)] for x,y in pts]})
                else:
                    legacy_pts = data.get("points", None)
                    if legacy_pts and len(legacy_pts) >= 3:
                        aois = [{"name": "AOI 1",
                                 "polygon": [[float(x),float(y)] for x,y in legacy_pts]}]
                if img_name in by_name and aois:
                    self.aoi_map[str(by_name[img_name])] = aois
                    loaded += 1
            except Exception as e:
                self._log(f"[AOI] import failed for {jf}: {e}")
        if loaded and not silent:
            self._log(f"[AOI] Imported AOIs for {loaded} images from {folder}")
        return loaded

    def _export_aois_to_input(self, imgs, force=False):
        if not imgs: return 0
        if not (force or (self.use_aoi.get() and self.aoi_map)): return 0
        root = imgs[0].parent if self.selected_files else Path(self.input_dir.get().strip())
        (root / "aoi").mkdir(parents=True, exist_ok=True)
        if cv2 is not None: (root / "aoi_masks").mkdir(parents=True, exist_ok=True)

        saved = 0
        for p in imgs:
            aois = self.aoi_map.get(str(p), [])
            if not aois: continue
            out = {
                "image": p.name,
                "aois": [{"name": a.get("name","AOI"),
                          "polygon": [list(pt) for pt in a.get("polygon", a.get("pts", []))],
                          "points":  [list(pt) for pt in a.get("polygon", a.get("pts", []))]}  # legacy mirror
                         for a in aois]
            }
            with open(root/"aoi"/f"{p.stem}.json","w",encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            if cv2 is not None:
                im = cv2.imread(str(p))
                if im is not None:
                    h,w = im.shape[:2]
                    m = build_union_mask(h, w, aois)
                    if m is not None:
                        cv2.imwrite(str(root/"aoi_masks"/f"{p.stem}.png"), m)
            saved += 1
        self._log(f"[AOI] Persisted to INPUT/aoi & aoi_masks ({saved} images)")
        return saved

    # ---------- UI helpers ----------
    def _update_preset_label(self):
        snap, name = _quality_to_name_snapped(self.quality.get())
        p = BUILTIN_PRESETS[name]
        try:
            self.preset_label.config(
                text=f"{name} • tile={p['tile']}  overlap={p['overlap']}  conf={p['conf']}  nms={p['iou_nms']}  WBF={p['use_wbf']}"
            )
        except Exception:
            pass

    def open_advanced(self):
        open_advanced(self)

    def _resolve_inputs(self) -> list[Path]:
        if self.selected_files:
            return self.selected_files[:]
        if not self.input_dir.get().strip():
            return []
        return collect_images(Path(self.input_dir.get().strip()))

    # ===== classes grid =====
    def _populate_classes(self, id2name: dict[int, str] | list[str]):
        if not hasattr(self, "classes_container") or self.classes_container is None:
            return
        container = self.classes_container
        prev = set(cid for (_nm, var, cid) in self.class_vars if var.get())
        for w in container.winfo_children():
            try: w.destroy()
            except: pass
        self.class_vars.clear()
        if isinstance(id2name, dict):
            pairs = [(cid, nm) for cid, nm in sorted(id2name.items(), key=lambda kv: kv[0])]
        else:
            pairs = list(enumerate(id2name))
        try:
            avail = max(320, int(self.classes_scroll.canvas.winfo_width()))
        except Exception:
            avail = 800
        cols = int(round(avail / float(self.CLASS_CELL_PX)))
        cols = max(self.CLASS_COL_MIN, min(self.CLASS_COL_MAX, cols))
        self._class_cols = cols
        col_w = max(120, (avail // cols))
        for c in range(cols):
            container.grid_columnconfigure(c, minsize=col_w, weight=1, uniform="classes")
        for idx, (cid, nm) in enumerate(pairs):
            var = tk.BooleanVar(value=(cid in prev))
            r, c = divmod(idx, cols)
            cell = tk.Frame(container, width=col_w)
            cell.grid(row=r, column=c, sticky="nw", padx=6, pady=3)
            cell.grid_propagate(False)
            tk.Checkbutton(cell, text=f"[{cid}] {nm}", variable=var, anchor="w",
                           justify="left", wraplength=col_w-12).pack(fill="x", expand=True, anchor="w")
            self.class_vars.append((nm, var, cid))

    def _on_classes_canvas_config(self, width: int):
        try:
            avail = max(320, int(width))
        except Exception:
            try: avail = max(320, int(self.classes_scroll.canvas.winfo_width()))
            except Exception: avail = 800
        new_cols = int(round(avail / float(self.CLASS_CELL_PX)))
        new_cols = max(self.CLASS_COL_MIN, min(self.CLASS_COL_MAX, new_cols))
        if new_cols != self._class_cols and self.class_names:
            self._class_cols = new_cols
            self._populate_classes(self.class_names)
        else:
            col_w = max(120, (avail // max(1, self._class_cols)))
            for c in range(self._class_cols):
                try:
                    self.classes_container.grid_columnconfigure(c, minsize=col_w, weight=1, uniform="classes")
                except Exception:
                    pass

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
            self.log.insert("end", line + "\n")
            self.log.see("end")
        except Exception:
            pass
        try:
            if self._logfile:
                with open(self._logfile, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass

    # ---------- progress smoothing ----------
    def _smooth_to(self, target_pct: float):
        self._smooth_target = max(0.0, min(100.0, float(target_pct)))
        if self._smooth_job is None:
            self._smooth_job = self.after(50, self._smooth_tick)

    def _smooth_tick(self):
        cur = float(self.progress_var.get() or 0.0)
        tgt = float(self._smooth_target)
        if abs(tgt - cur) <= 1.0:
            self.progress_var.set(tgt)
            self._smooth_job = None
        else:
            step = 1.0 if tgt > cur else -1.0
            self.progress_var.set(cur + step)
            self._smooth_job = self.after(50, self._smooth_tick)

    # ---------- threaded run ----------
    def start(self):
        imgs = self._resolve_inputs()
        if not imgs:
            messagebox.showerror("Input", "Select a valid image folder or files."); return
        model = self.weights_path.get().strip()
        if not model:
            messagebox.showerror("Model", "Select a valid weights file (.pt or .onnx)."); return

        imported = self._import_aois_from_folder_for_images(imgs, silent=True)
        if imported:
            self._log(f"[AOI] Reused existing AOIs for {imported} images.")
        # NOTE: Do NOT auto-enable AOI here. Respect the toggle fully.

        if self.class_names and not self._selected_classes():
            messagebox.showwarning("Classes", "Select at least one class."); return

        if self.use_aoi.get() and self.require_aoi_all.get():
            missing = [p for p in imgs if str(p) not in self.aoi_map or not self.aoi_map[str(p)]]
            if missing:
                messagebox.showwarning("AOI", f"AOI missing for {len(missing)} images."); return

        # outdir: prefer user field; otherwise fall back to ./output (not ./results)
        outdir = Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else (
            Path(__file__).parent.resolve() / "output"
        )
        self._set_logfile(outdir)
        self._log(f"Output → {outdir}")
        self._log(f"Using model: {model}  | engine={self.engine_var.get()} device={self.device_var.get()}")

        snap, qname = _quality_to_name_snapped(self.quality.get())
        base = BUILTIN_PRESETS[qname].copy()
        if self.advanced_override and self.advanced_params:
            base.update(self.advanced_params)

        self._export_aois_to_input(imgs, force=False)

        use_legacy_pt = (self.engine_var.get() in ("auto", "pt") and model.lower().endswith(".pt"))

        self.btn_start.config(state="disabled")
        self.btn_abort.config(state="normal")
        self._stop = False
        self.progress_var.set(0.0)
        self.update_idletasks()

        def work():
            try:
                if use_legacy_pt:
                    ## --- SINGLE RUN, tolerant to both return shapes ---
                    result = run_legacy_pt(
                        imgs=imgs, outdir=outdir, model_path=model,
                        tile=int(base["tile"]), overlap=float(base["overlap"]),
                        conf=float(base["conf"]), iou=float(base["iou_nms"]),
                        selected_classes=self._selected_classes(),
                        overlay_mode=self.overlay_mode.get(),
                        draw_centroid=bool(self.draw_centroid.get()),
                        ## AOI OFF must also set mode='off' to avoid "exclude all" behavior
                        aoi_mode=("off" if not self.use_aoi.get() else self.aoi_mode.get()),
                        aoi_box_frac=float(self.aoi_box_frac.get()),
                        aoi_map=(self.aoi_map if self.use_aoi.get() else {}),
                        ## Ignore progress updates once abort flag is set
                        progress_cb=lambda pct, txt: self._tsafe(
                            lambda: (None if self._stop else self._smooth_to(pct),
                                     None if self._stop else self.progress_label.set(txt))
                        ),
                        stop_cb=lambda: self._stop,
                        class_id_to_name=(self.class_names or None),
                        logger=self._log,
                        return_dets=True,   # request (totals, dets_map)
                    )

                    dets_map = {}
                    totals = {}
                    if isinstance(result, tuple) and len(result) == 2:
                        totals, dets_map = result
                    else:
                        totals = result if isinstance(result, dict) else {}

                    self._maybe_export_geojson(imgs, outdir, dets_map)

                else:
                    totals = self._run_engine_core(imgs, outdir, model, base, qname)

                self._tsafe(lambda: self.progress_label.set(f"Done. Output: {outdir}"))
                self._tsafe(lambda: self._log(f"Done. Totals: {totals}"))
                self._tsafe(lambda: messagebox.showinfo("Done", f"Processed {len(imgs)} images.\nSaved to: {outdir}"))
            except Exception as e:
                err = str(e)
                tb = traceback.format_exc()
                def report():
                    ## Treat abort as clean cancel, not an error popup
                    if isinstance(e, KeyboardInterrupt) or "ABORT" in err.upper():
                        self.progress_label.set("Aborted.")
                        self._log("Aborted by user.")
                        return
                    messagebox.showerror("Error", err)
                    self._log(f"[ERROR] {err}")
                    self._log(tb)
                self._tsafe(report)
            finally:
                self._tsafe(lambda: (self.btn_start.config(state="normal"), self.btn_abort.config(state="disabled")))
        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def abort(self):
        ## Hard abort: set flag, cancel smoother, freeze UI immediately
        self._stop = True
        try:
            if self._smooth_job is not None:
                self.after_cancel(self._smooth_job)
                self._smooth_job = None
        except Exception:
            pass
        try:
            self.progress_label.set("Aborting…")
        except Exception:
            pass
        self._log("=== ABORT requested ===")

    def _tsafe(self, fn):
        try: self.after(0, fn)
        except Exception: pass

    # ---------- class toggles ----------
    def select_all_classes(self, state: bool = True):
        for (_nm, var, _cid) in self.class_vars:
            try:
                var.set(bool(state))
            except Exception:
                pass

    def invert_classes(self):
        for (_nm, var, _cid) in self.class_vars:
            try:
                var.set(not bool(var.get()))
            except Exception:
                pass

    # ---------- engine-core path ----------
    def _run_engine_core(self, imgs, outdir: Path, model, base, qname: str):
        cfg = InferConfig(
            model_path=model, engine=self.engine_var.get(), device=self.device_var.get(),
            conf=float(base["conf"]), iou=float(base["iou_nms"]), imgsz=int(base["tile"]),
            classes=self._selected_classes(),
            aoi_mode=("off" if not self.use_aoi.get() else self.aoi_mode.get()),  ## mirror AOI OFF here too
            aoi_box_frac=float(self.aoi_box_frac.get()),
            annotate=bool(self.annotate.get()), draw_centroid=bool(self.draw_centroid.get()),
            use_tiling=True, tile=int(base["tile"]), overlap=float(base["overlap"]),
            use_wbf=bool(base.get("use_wbf", True)),
            wbf_iou=float(base.get("wbf_iou", auto_wbf_iou(qname, base["iou_nms"])) if base.get("wbf_iou") not in (None, "") else auto_wbf_iou(qname, base["iou_nms"])),
            wbf_alpha=float(base.get("wbf_alpha", 0.20)),
            seam_band_factor=float(base.get("seam_band_factor", 0.10)),
            seam_weight=float(base.get("seam_weight", 0.35)),
            overlay_mode=self.overlay_mode.get(), persist_aoi_to_input=True,
        )
        engine = ModelEngine(cfg)
        self._log(f"[engine] names={engine.available_classes()}")

        def pcb(i, n, eta_sec):
            if self._stop:
                return
            frac = i / max(1, n)
            m = int(eta_sec // 60); s = int(eta_sec % 60)
            self._smooth_to(frac * 100.0)
            self.progress_label.set(f"Image {i}/{n} — ETA {m:02d}:{s:02d}")

        dets_map = None
        try:
            per_image, totals, dets_map = engine.predict_batch(
                imgs, aoi_map=self.aoi_map if self.use_aoi.get() else {},
                outdir=outdir, annotate=cfg.annotate,
                progress_cb=pcb, abort_cb=lambda: self._stop, return_dets=True
            )
        except TypeError:
            per_image, totals = engine.predict_batch(
                imgs, aoi_map=self.aoi_map if self.use_aoi.get() else {},
                outdir=outdir, annotate=cfg.annotate,
                progress_cb=pcb, abort_cb=lambda: self._stop
            )

        class_names = sorted({k for _, cnt in per_image for k in cnt.keys()})
        rows = [[path] + [cnt.get(c, 0) for c in class_names] for path, cnt in per_image]
        save_csv(rows, header=["image_path"] + class_names, out_path=outdir / "results_per_image.csv")
        save_json(totals, out_path=outdir / "results_totals.json")
        save_run_metadata(outdir, imgs, cfg, totals)

        self._maybe_export_geojson(imgs, outdir, dets_map)
        return totals

    # ---------- Geo export helper ----------
    def _maybe_export_geojson(self, imgs, outdir: Path, dets_map):
        if export_geojson_for_image is None:
            self._log("[GEO] geo_export.py not found — skipping Geo export."); return
        if not dets_map:
            self._log("[GEO] No detection details provided by engine — skipping Geo export."); return

        gis_dir = outdir / "gis"
        try:
            gis_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        for p in imgs:
            try:
                dets = self._normalize_dets(dets_map.get(str(p)) or dets_map.get(p) or [])
                # pass AOI list as-is; exporter can write AOIs too if desired
                aois = self.aoi_map.get(str(p), [])
                geo = export_geojson_for_image(p, dets, aois, out_dir=gis_dir, crs_hint=None)
                if geo: self._log(f"[GEO] Wrote {geo.name}")
            except Exception as e:
                self._log(f"[GEO] export failed for {Path(p).name}: {e}")

    @staticmethod
    def _normalize_dets(raw_list):
        norm = []
        for d in (raw_list or []):
            try:
                cls = d.get("cls") or d.get("class") or d.get("label")
                conf = float(d.get("conf", d.get("confidence", 0.0)))
                bbox = d.get("bbox") or d.get("box") or d.get("xyxy")
                if not bbox or len(bbox) != 4: 
                    continue
                x1,y1,x2,y2 = [float(v) for v in bbox]
                cx, cy = d.get("centroid", ((x1+x2)/2.0, (y1+y2)/2.0))
                norm.append({"cls": cls, "conf": conf, "bbox": [x1,y1,x2,y2], "centroid": [float(cx),float(cy)]})
            except Exception:
                continue
        return norm

    def _selected_classes(self):
        return [cid for (_nm, v, cid) in self.class_vars if v.get()]

if __name__ == "__main__":
    App().mainloop()
