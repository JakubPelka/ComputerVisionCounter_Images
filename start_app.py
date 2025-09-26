# start_app.py — main app entry for ComputerVision Counter
# - Auto engine/device (no selectors in UI)
# - AOI editor persists AOIs and auto-enables Use AOI
# - AOI import/export compatibility (old 'points' and new 'polygon')
# - Class grid keeps REAL class IDs
# - Smooth progress updates

from __future__ import annotations
import sys, json, time, threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# --- ensure local pkgs/ are first on sys.path
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

# --- project imports
from app_core import (
    InferConfig, ModelEngine, collect_images,
    save_csv, save_json, save_run_metadata
)
from widgets import ScrollableFrame, AOIEditor
import ui_panels
from ui_advanced import open_advanced
from legacy_pt_runner import run_legacy_pt, build_union_mask

APP_TITLE = "ComputerVision Counter — Count anything without coding"

# -------- presets / defaults --------
QUALITY_PRESETS = {
    1: {"tile": 640,  "overlap": 0.15, "conf": 0.40, "iou_nms": 0.65, "use_wbf": True},
    2: {"tile": 896,  "overlap": 0.25, "conf": 0.45, "iou_nms": 0.60, "use_wbf": True},
    3: {"tile": 1024, "overlap": 0.30, "conf": 0.50, "iou_nms": 0.55, "use_wbf": True},
    4: {"tile": 1280, "overlap": 0.45, "conf": 0.60, "iou_nms": 0.50, "use_wbf": True},
    5: {"tile": 2560, "overlap": 0.60, "conf": 0.75, "iou_nms": 0.40, "use_wbf": True},
}
DEFAULT_QUALITY = 5
DEFAULT_WBF_ALPHA = 0.20
DEFAULT_SEAM_IOU_LOW = 0.30
DEFAULT_SEAM_BAND_FACTOR = 0.10
DEFAULT_SEAM_WEIGHT = 0.35
DEFAULT_MARGIN_WEIGHT = 0.25

def auto_wbf_iou(q, nms):  # helper for a decent default WBF IoU
    return 0.60 if int(q) == 5 else max(0.55, float(nms))


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
        self.input_dir = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value="")
        self.weights_path = tk.StringVar(value="")
        self.selected_files: list[Path] = []

        # “hidden” advanced flags (kept in code, not shown in UI)
        self.engine_var = tk.StringVar(value="auto")  # auto/pt/onnx
        self.device_var = tk.StringVar(value="auto")  # auto/cpu/cuda:0

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
        self.aoi_mode = tk.StringVar(value="centroid")  # centroid | box
        self.aoi_box_frac = tk.DoubleVar(value=0.20)
        # image path (string) -> list of {"name": str, "polygon": [[x,y],...]}
        self.aoi_map: dict[str, list[dict]] = {}

        # visualization
        self.overlay_mode = tk.StringVar(value="boxes_conf")  # boxes | boxes_conf | centroid
        self.annotate = tk.BooleanVar(value=True)
        self.draw_centroid = tk.BooleanVar(value=False)

        # classes
        self.class_names: dict[int, str] = {}                          # id -> name
        self.class_vars: list[tuple[str, tk.BooleanVar, int]] = []     # (name, var, class_id)
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

        # build UI
        ui_panels.build_main_ui(self)

        # hotkeys
        self.bind("<Control-Return>", lambda e: self.start())
        self.bind("<Control-a>", lambda e: self._open_aoi_editor())

        self._log("Ready.")

    # ---------- pickers ----------
    def browse_input(self):
        d = filedialog.askdirectory(title="Select input folder with images")
        if not d:
            return
        self.input_dir.set(d)
        imgs = collect_images(Path(d))
        self.progress_label.set(f"{len(imgs)} images ready")
        self._refresh_files_label()

    def browse_files(self):
        files = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")]
        )
        if not files:
            return
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
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_dir.set(d)

    def browse_weights(self):
        p = filedialog.askopenfilename(
            title="Select weights (.pt or .onnx)",
            filetypes=[("Models","*.pt *.onnx"), ("All files","*.*")]
        )
        if not p:
            return
        self.weights_path.set(p)
        self._log(f"Model: {Path(p).name}")
        # preload classes if engine can provide them
        try:
            cfg = InferConfig(model_path=p, engine=self.engine_var.get(), device=self.device_var.get())
            eng = ModelEngine(cfg)
            self.class_names = eng.available_classes()  # dict: id->name
            self._populate_classes(self.class_names)    # keep real IDs
            self._log(f"Loaded {len(self.class_names)} classes from engine.")
        except Exception as e:
            self._log(f"[WARN] Could not read classes from engine: {e}")

    # ---------- AOI: helpers / actions ----------
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

        # if AOIs exist on disk for these images, preload them (silent)
        self._import_aois_from_folder_for_images(imgs, silent=True)

        idx = 0
        top = tk.Toplevel(self)
        top.title("AOI — define polygons per image")
        top.geometry("1100x840")
        # --- keep the editor above the main window on open
        try:
            top.transient(self)
            top.lift()
            top.attributes("-topmost", True)
            top.after(250, lambda: top.attributes("-topmost", False))
        except Exception:
            pass

        def save_current():
            p = imgs[idx]
            self.aoi_map[str(p)] = editor.get_aois()

        def on_change(_aois):
            save_current()

        # nav row
        nav = tk.Frame(top); nav.pack(fill="x", pady=4)
        idx_var = tk.StringVar(value=f"1/{len(imgs)}")
        tk.Button(nav, text="⟵ Prev", command=lambda: load_idx(idx-1)).pack(side="left")
        tk.Button(nav, text="Next ⟶", command=lambda: load_idx(idx+1)).pack(side="left", padx=6)
        tk.Label(nav, textvariable=idx_var).pack(side="left", padx=10)
        tk.Label(nav, text="(Finish polygon: Ctrl+Enter — Undo vertex: Ctrl+Backspace)", fg="#666").pack(side="left", padx=10)

        # editor
        holder = tk.Frame(top); holder.pack(fill="both", expand=True)
        editor = AOIEditor(holder, on_change=on_change); editor.pack(fill="both", expand=True)

        def load_idx(i):
            nonlocal idx
            save_current()
            i = max(0, min(i, len(imgs)-1)); idx = i
            p = imgs[i]
            editor.load_image(str(p))
            aois = self.aoi_map.get(str(p), None)
            editor.set_aois(aois if aois is not None else [])
            idx_var.set(f"{i+1}/{len(imgs)}")

        def on_close():
            save_current()
            # persist to <input>/aoi & aoi_masks immediately
            self._export_aois_to_input(imgs, force=True)
            # enable Use AOI if any AOIs exist
            if any(self.aoi_map.get(str(p)) for p in imgs):
                self.use_aoi.set(True)
                self._log("[AOI] AOIs saved; Use AOI enabled.")
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", on_close)
        load_idx(0)  # show first image immediately

    def import_aois_from_input(self):
        imgs = self._resolve_inputs()
        if not imgs:
            messagebox.showinfo("AOI", "Select input images first.")
            return
        loaded = self._import_aois_from_folder_for_images(imgs, silent=False)
        if loaded > 0:
            self.use_aoi.set(True)
            messagebox.showinfo("AOI", f"Imported AOIs for {loaded} images.")

    def export_aois_now(self):
        imgs = self._resolve_inputs()
        if not imgs:
            messagebox.showinfo("AOI", "Select input images first.")
            return
        count = self._export_aois_to_input(imgs, force=True)
        messagebox.showinfo("AOI", f"Exported AOIs for {count} images.")

    # --- core import/export helpers
    def _import_aois_from_folder_for_images(self, imgs, silent: bool):
        """Read AOIs from <input>/aoi/*.json for the given image list (supports 'points' and 'polygon')."""
        if not imgs:
            return 0
        root = imgs[0].parent if self.selected_files else Path(self.input_dir.get().strip())
        folder = root / "aoi"
        if not folder.exists():
            if not silent:
                self._log(f"[AOI] No 'aoi' folder under {root}")
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
        """Persist AOIs JSON + union masks next to input images. Returns number saved."""
        if not imgs:
            return 0
        if not (force or (self.use_aoi.get() and self.aoi_map)):
            return 0
        root = imgs[0].parent if self.selected_files else Path(self.input_dir.get().strip())
        (root / "aoi").mkdir(parents=True, exist_ok=True)
        if cv2 is not None:
            (root / "aoi_masks").mkdir(parents=True, exist_ok=True)

        saved = 0
        for p in imgs:
            aois = self.aoi_map.get(str(p), [])
            if not aois:
                continue
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
                    m = build_union_mask(h, w, aois)  # expects 'polygon'
                    if m is not None:
                        cv2.imwrite(str(root/"aoi_masks"/f"{p.stem}.png"), m)
            saved += 1
        self._log(f"[AOI] Persisted to INPUT/aoi & aoi_masks ({saved} images)")
        return saved

    # ---------- UI helpers ----------
    def _update_preset_label(self):
        p = QUALITY_PRESETS.get(int(self.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY])
        try:
            self.preset_label.config(
                text=f"tile={p['tile']}  overlap={p['overlap']}  conf={p['conf']}  nms={p['iou_nms']}  WBF={p['use_wbf']}"
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

    # ===== Dynamic classes grid (keep REAL class IDs) =====
    def _populate_classes(self, id2name: dict[int, str] | list[str]):
        if not hasattr(self, "classes_container") or self.classes_container is None:
            return
        container = self.classes_container

        prev = set(cid for (_nm, var, cid) in self.class_vars if var.get())

        # clear UI
        for w in container.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        self.class_vars.clear()

        # normalize into list of (class_id, name)
        if isinstance(id2name, dict):
            pairs = [(cid, nm) for cid, nm in sorted(id2name.items(), key=lambda kv: kv[0])]
        else:
            pairs = list(enumerate(id2name))

        # compute columns based on available width
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

        # build cells
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
            try:
                avail = max(320, int(self.classes_scroll.canvas.winfo_width()))
            except Exception:
                avail = 800
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
            messagebox.showerror("Input", "Select a valid image folder or files.")
            return
        model = self.weights_path.get().strip()
        if not model:
            messagebox.showerror("Model", "Select a valid weights file (.pt or .onnx).")
            return

        # Load AOIs from disk if present so a second run reuses them automatically
        imported = self._import_aois_from_folder_for_images(imgs, silent=True)
        if imported:
            self._log(f"[AOI] Reused existing AOIs for {imported} images.")
        # If AOIs exist, auto-enable Use AOI
        if not self.use_aoi.get() and (any(self.aoi_map.get(str(p)) for p in imgs) or imported):
            self.use_aoi.set(True)
            self._log("[AOI] AOIs found — Use AOI enabled automatically.")

        # require at least one class if names are available
        if self.class_names and not self._selected_classes():
            messagebox.showwarning("Classes", "Select at least one class.")
            return

        if self.use_aoi.get() and self.require_aoi_all.get():
            missing = [p for p in imgs if str(p) not in self.aoi_map or not self.aoi_map[str(p)]]
            if missing:
                messagebox.showwarning("AOI", f"AOI missing for {len(missing)} images.")
                return

        outdir = Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else (
            (imgs[0].parent if self.selected_files else Path(self.input_dir.get().strip())) / "results"
        )
        self._set_logfile(outdir)
        self._log(f"Output → {outdir}")
        self._log(f"Using model: {model}  | engine={self.engine_var.get()} device={self.device_var.get()}")

        base = QUALITY_PRESETS.get(int(self.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY]).copy()
        if self.advanced_override and self.advanced_params:
            base.update(self.advanced_params)
            if base.get("wbf_auto", True):
                base["wbf_iou"] = auto_wbf_iou(int(self.quality.get()), base["iou_nms"])

        # Persist AOIs to input (always if any)
        self._export_aois_to_input(imgs, force=False)

        # choose path: legacy YOLOv5/8 .pt runner or core engine (inc. ONNX)
        use_legacy_pt = (self.engine_var.get() in ("auto", "pt") and model.lower().endswith(".pt"))

        # arm UI
        self.btn_start.config(state="disabled")
        self.btn_abort.config(state="normal")
        self._stop = False
        self.progress_var.set(0.0)
        self.update_idletasks()

        def work():
            try:
                if use_legacy_pt:
                    totals = run_legacy_pt(
                        imgs=imgs, outdir=outdir, model_path=model,
                        tile=int(base["tile"]), overlap=float(base["overlap"]),
                        conf=float(base["conf"]), iou=float(base["iou_nms"]),
                        selected_classes=self._selected_classes(),  # REAL IDs
                        overlay_mode=self.overlay_mode.get(),
                        draw_centroid=bool(self.draw_centroid.get()),
                        aoi_mode=self.aoi_mode.get(),
                        aoi_box_frac=float(self.aoi_box_frac.get()),
                        aoi_map=self.aoi_map if self.use_aoi.get() else {},
                        progress_cb=lambda pct, txt: self._tsafe(lambda: (self._smooth_to(pct), self.progress_label.set(txt))),
                        stop_cb=lambda: self._stop,
                        class_id_to_name=(self.class_names or None),
                        logger=self._log
                    )
                else:
                    totals = self._run_engine_core(imgs, outdir, model, base)

                self._tsafe(lambda: self.progress_label.set(f"Done. Output: {outdir}"))
                self._tsafe(lambda: self._log(f"Done. Totals: {totals}"))
                self._tsafe(lambda: messagebox.showinfo("Done", f"Processed {len(imgs)} images.\nSaved to: {outdir}"))
            except Exception as e:
                self._tsafe(lambda: (messagebox.showerror("Error", str(e)), self._log(f"[ERROR] {e}")))
            finally:
                self._tsafe(lambda: (self.btn_start.config(state="normal"), self.btn_abort.config(state="disabled")))
        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def abort(self):
        self._stop = True
        self._log("=== ABORT requested ===")

    def _tsafe(self, fn):
        try:
            self.after(0, fn)
        except Exception:
            pass

    # ---------- engine-core path ----------
    def _run_engine_core(self, imgs, outdir: Path, model, base):
        cfg = InferConfig(
            model_path=model, engine=self.engine_var.get(), device=self.device_var.get(),
            conf=float(base["conf"]), iou=float(base["iou_nms"]), imgsz=int(base["tile"]),
            classes=self._selected_classes(),  # REAL IDs
            aoi_mode=self.aoi_mode.get(), aoi_box_frac=float(self.aoi_box_frac.get()),
            annotate=bool(self.annotate.get()), draw_centroid=bool(self.draw_centroid.get()),
            use_tiling=True, tile=int(base["tile"]), overlap=float(base["overlap"]),
            use_wbf=bool(base.get("use_wbf", True)),
            wbf_iou=float(base.get("wbf_iou", auto_wbf_iou(int(self.quality.get()), base["iou_nms"]))),
            wbf_alpha=float(base.get("wbf_alpha", DEFAULT_WBF_ALPHA)),
            seam_band_factor=float(base.get("seam_band_factor", DEFAULT_SEAM_BAND_FACTOR)),
            seam_weight=float(base.get("seam_weight", DEFAULT_SEAM_WEIGHT)),
            overlay_mode=self.overlay_mode.get(), persist_aoi_to_input=True,
        )
        engine = ModelEngine(cfg)
        self._log(f"[engine] names={engine.available_classes()}")

        def pcb(i, n, eta_sec):
            frac = i / max(1, n)
            m = int(eta_sec // 60); s = int(eta_sec % 60)
            self._smooth_to(frac * 100.0)
            self.progress_label.set(f"Image {i}/{n} — ETA {m:02d}:{s:02d}")

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
        return totals

    # ---------- classes helpers ----------
    def _selected_classes(self):
        return [cid for (_nm, v, cid) in self.class_vars if v.get()]


if __name__ == "__main__":
    App().mainloop()
