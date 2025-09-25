# start_app.py — split launcher (1:1 behavior), UI in ui_panels.py (English)
from __future__ import annotations

import os, sys, time, errno
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# Prefer local vendored packages
def _add_local_pkgs():
    here = str(Path(__file__).parent.resolve())
    for d in ["pkgs", "_pkgs"]:
        cand = os.path.join(here, d)
        if os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
_add_local_pkgs()

# Core logic and engine wrappers (already extracted)
from app_core import (
    InferConfig, ModelEngine, collect_images,
    save_csv, save_json, save_run_metadata
)
# Small UI widgets
from widgets import ScrollableFrame, AOIEditor

# UI layout is in a separate module
import ui_panels

APP_TITLE = "ComputerVision Counter — Count anything without coding"

# ===== Presets & constants (preserved spirit of backup) =====
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

def auto_wbf_iou(quality: int, iou_nms: float) -> float:
    return 0.60 if int(quality) == 5 else max(0.55, float(iou_nms))

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1120x880")
        self.minsize(980, 760)

        # Expose for ui_panels
        self.ScrollableFrame = ScrollableFrame

        # --- Paths / selections ---
        self.input_dir = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value="")
        self.weights_path = tk.StringVar(value="")
        self.selected_files: list[Path] = []

        # --- Engine & device (pt/onnx/auto; 0/cpu/auto) ---
        self.engine_var = tk.StringVar(value="auto")
        self.device_var = tk.StringVar(value="auto")

        # --- Quality & advanced ---
        self.quality = tk.IntVar(value=DEFAULT_QUALITY)
        self.advanced_override = False
        self.advanced_params = {
            "tile": QUALITY_PRESETS[DEFAULT_QUALITY]["tile"],
            "overlap": QUALITY_PRESETS[DEFAULT_QUALITY]["overlap"],
            "conf": QUALITY_PRESETS[DEFAULT_QUALITY]["conf"],
            "iou_nms": QUALITY_PRESETS[DEFAULT_QUALITY]["iou_nms"],
            "use_wbf": True,
            "wbf_alpha": DEFAULT_WBF_ALPHA,
            "wbf_iou": None,        # None => auto
            "wbf_auto": True,
            "seam_iou_low": DEFAULT_SEAM_IOU_LOW,
            "seam_band_factor": DEFAULT_SEAM_BAND_FACTOR,
            "seam_weight": DEFAULT_SEAM_WEIGHT,
            "margin_weight": DEFAULT_MARGIN_WEIGHT,
        }

        # --- AOI settings ---
        self.use_aoi = tk.BooleanVar(value=False)
        self.require_aoi_all = tk.BooleanVar(value=False)    # ensure exists for UI
        self.persist_aoi = tk.BooleanVar(value=True)
        self.aoi_mode = tk.StringVar(value="centroid")       # 'centroid' | 'box'
        self.aoi_box_frac = tk.DoubleVar(value=0.20)
        self.aoi_map: dict[str, list[dict]] = {}             # image_path -> list of AOIs (polygons)

        # --- Visualization / annotation ---
        self.overlay_mode = tk.StringVar(value="boxes_conf")  # 'boxes'|'boxes_conf'|'centroid'
        self.annotate = tk.BooleanVar(value=True)
        self.draw_centroid = tk.BooleanVar(value=False)

        # --- Classes ---
        self.show_class_checkboxes = tk.BooleanVar(value=True)
        self.require_class_selection = tk.BooleanVar(value=False)
        self.class_names: dict[int,str] = {}  # id -> name
        self.class_vars: dict[int, tk.BooleanVar] = {}

        # --- Progress / abort ---
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label = tk.StringVar(value="Ready.")
        self._stop = False

        # Build UI
        ui_panels.build_main_ui(self)

        # hotkeys
        self.bind("<Control-Return>", lambda e: self.start())
        self.bind("<Control-a>", lambda e: self._open_aoi_editor())

        self._log("Ready.")

    # ---------------- File pickers ----------------
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
            filetypes=[("Models", "*.pt *.onnx"), ("All files", "*.*")]
        )
        if not p: return
        self.weights_path.set(p)
        self._log(f"Model: {Path(p).name}")
        # Try load class list
        try:
            cfg = InferConfig(model_path=p, engine=self.engine_var.get(), device=self.device_var.get())
            eng = ModelEngine(cfg)
            self.class_names = eng.available_classes()
            self._build_class_checkboxes()
            self._log(f"Loaded {len(self.class_names)} classes.")
        except Exception as e:
            self._log(f"[WARN] Failed to read classes: {e}")

    # ---------------- AOI ----------------
    def _on_toggle_use_aoi(self):
        if self.use_aoi.get():
            self._open_aoi_editor()

    def _open_aoi_editor(self):
        # Multi-image AOI editor using widgets.AOIEditor (supports multi AOIs w/ names per your new widgets)
        imgs = self._resolve_inputs()
        if not imgs:
            messagebox.showinfo("AOI", "Select input images first.")
            return
        # Open one-by-one editor with navigation
        idx = 0
        top = tk.Toplevel(self); top.title("AOI — define polygons per image"); top.geometry("1100x840")
        nav = tk.Frame(top); nav.pack(fill="x", pady=4)
        idx_var = tk.StringVar(value=f"1/{len(imgs)}")
        tk.Button(nav, text="⟵ Prev", command=lambda: load_idx(idx-1)).pack(side="left")
        tk.Button(nav, text="Next ⟶", command=lambda: load_idx(idx+1)).pack(side="left", padx=6)
        tk.Label(nav, textvariable=idx_var).pack(side="left", padx=10)
        tk.Button(nav, text="Save AOI for current", command=lambda: save_current()).pack(side="left", padx=10)

        holder = tk.Frame(top); holder.pack(fill="both", expand=True)
        editor = AOIEditor(holder)  # load/set/get API
        editor.pack(fill="both", expand=True)

        def load_idx(i):
            nonlocal idx
            i = max(0, min(i, len(imgs)-1))
            idx = i
            p = imgs[i]
            editor.load_image(str(p))
            aois = self.aoi_map.get(str(p), [])
            if aois:
                editor.set_aois(aois)
            idx_var.set(f"{i+1}/{len(imgs)}")

        def save_current():
            p = imgs[idx]
            self.aoi_map[str(p)] = editor.get_aois()
            self._log(f"[AOI] Saved {len(self.aoi_map[str(p)])} polygons for {Path(p).name}")

        load_idx(0)

    # ---------------- Advanced / preset hint ----------------
    def _update_preset_label(self):
        p = QUALITY_PRESETS.get(int(self.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY])
        try:
            self.preset_label.config(
                text=f"tile={p['tile']}  overlap={p['overlap']}  conf={p['conf']}  nms={p['iou_nms']}  WBF={p['use_wbf']}"
            )
        except Exception:
            pass

    def open_advanced(self):
        ui_panels.open_advanced(self)

    # ---------------- Helpers ----------------
    def _resolve_inputs(self) -> list[Path]:
        if self.selected_files:
            return self.selected_files[:]
        if not self.input_dir.get().strip():
            return []
        return collect_images(Path(self.input_dir.get().strip()))

    def _build_class_checkboxes(self):
        # Clear container
        for w in getattr(self, "classes_container", []).winfo_children():
            try: w.destroy()
            except: pass
        self.class_vars.clear()
        if not self.class_names:
            tk.Label(self.classes_container, text="(Load weights to show classes)").grid(row=0, column=0, sticky="w")
            return
        r = 0
        for i, name in sorted(self.class_names.items(), key=lambda kv: kv[0]):
            var = tk.BooleanVar(value=False)  # explicit selection
            self.class_vars[i] = var
            cb = tk.Checkbutton(self.classes_container, text=f"[{i}] {name}", variable=var, anchor="w")
            cb.grid(row=r//4, column=r%4, sticky="w", padx=6, pady=4)  # 4 columns
            r += 1

    def _selected_classes(self):
        if not self.show_class_checkboxes.get() or not self.class_vars:
            return None
        sel = [i for i,v in self.class_vars.items() if v.get()]
        return sel if sel else None

    # ---------------- Run / main loop ----------------
    def start(self):
        try:
            imgs = self._resolve_inputs()
            if not imgs:
                messagebox.showerror("Input", "Select a valid image folder or files.")
                return
            model = self.weights_path.get().strip()
            if not model:
                messagebox.showerror("Model", "Select a valid weights file (.pt or .onnx).")
                return
            if self.require_class_selection.get() and not self._selected_classes():
                messagebox.showwarning("Classes", "Select at least one class or disable the requirement.")
                return
            if self.use_aoi.get() and self.require_aoi_all.get():
                missing = [p for p in imgs if str(p) not in self.aoi_map or not self.aoi_map[str(p)]]
                if missing:
                    messagebox.showwarning("AOI", f"AOI missing for {len(missing)} images. Open AOI editor to set polygons.")
                    return

            # Output folder
            outdir = Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else (
                (imgs[0].parent if self.selected_files else Path(self.input_dir.get().strip())) / "results"
            )
            outdir.mkdir(parents=True, exist_ok=True)

            # Build params
            base = QUALITY_PRESETS.get(int(self.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY]).copy()
            if self.advanced_override and self.advanced_params:
                base.update(self.advanced_params)
                if base.get("wbf_auto", True):
                    base["wbf_iou"] = auto_wbf_iou(int(self.quality.get()), base["iou_nms"])

            cfg = InferConfig(
                model_path=model,
                engine=self.engine_var.get(),
                device=self.device_var.get(),
                conf=float(base["conf"]),
                iou=float(base["iou_nms"]),
                imgsz=int(base["tile"]),
                classes=self._selected_classes(),
                aoi_mode=self.aoi_mode.get(),
                aoi_box_frac=float(self.aoi_box_frac.get()),
                annotate=bool(self.annotate.get()),
                draw_centroid=bool(self.draw_centroid.get()),
                use_tiling=True,
                tile=int(base["tile"]),
                overlap=float(base["overlap"]),  # fraction 0..1
                use_wbf=bool(base.get("use_wbf", True)),
                wbf_iou=float(base.get("wbf_iou", auto_wbf_iou(int(self.quality.get()), base["iou_nms"]))),
                wbf_alpha=float(base.get("wbf_alpha", DEFAULT_WBF_ALPHA)),
                seam_band_factor=float(base.get("seam_band_factor", DEFAULT_SEAM_BAND_FACTOR)),
                seam_weight=float(base.get("seam_weight", DEFAULT_SEAM_WEIGHT)),
                overlay_mode=self.overlay_mode.get(),
                persist_aoi_to_input=bool(self.persist_aoi.get()),
            )
            engine = ModelEngine(cfg)

            # Persist AOIs if requested (guard WinError 183)
            if self.use_aoi.get() and self.aoi_map and self.persist_aoi.get():
                try:
                    engine.save_aoi_persistence(imgs, self.aoi_map)
                except OSError as e:
                    if getattr(e, "winerror", None) == 183 or e.errno == errno.EEXIST:
                        self._log("[AOI] AOI folders already exist. Continuing.")
                    else:
                        raise

            # Run
            self.btn_start.config(state="disabled")
            self.btn_abort.config(state="normal")
            self._stop = False
            self.progress_var.set(0.0); self.update_idletasks()

            per_image, totals = engine.predict_batch(
                imgs,
                aoi_map=self.aoi_map if self.use_aoi.get() else {},
                outdir=outdir,
                annotate=cfg.annotate,
                progress_cb=self._on_progress,
                abort_cb=lambda: self._stop
            )

            # Save tables
            class_names = sorted({k for _, cnt in per_image for k in cnt.keys()})
            rows = []
            for path, cnt in per_image:
                row = [path] + [cnt.get(c, 0) for c in class_names]
                rows.append(row)
            save_csv(rows, header=["image_path"] + class_names, out_path=outdir / "results_per_image.csv")
            save_json(totals, out_path=outdir / "results_totals.json")
            save_run_metadata(outdir, imgs, cfg, totals)

            self.progress_label.set(f"Done. Output: {outdir}")
            self._log(f"Done. Output: {outdir}")
            messagebox.showinfo("Done", f"Processed {len(per_image)} images.\nSaved to: {outdir}")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._log(f"[ERROR] {e}")
        finally:
            self.btn_start.config(state="normal")
            self.btn_abort.config(state="disabled")

    def abort(self):
        self._stop = True
        self._log("=== ABORTED by user ===")

    def _on_progress(self, i, n, eta_sec):
        pct = 100.0 * (i / max(1, n))
        self.progress_var.set(pct)
        m = int(eta_sec // 60); s = int(eta_sec % 60)
        self.progress_label.set(f"Image {i}/{n} — ETA {m:02d}:{s:02d}")
        self.update_idletasks()

    # logging
    def _log(self, msg: str):
        try:
            self.log.insert("end", msg + "\n")
            self.log.see("end")
        except Exception:
            print(msg)

if __name__ == "__main__":
    App().mainloop()