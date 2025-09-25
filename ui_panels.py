# ui_panels.py — UI builders split from start_app_backup (English text, same layout/controls)
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

def build_main_ui(app):
    frm = tk.Frame(app); frm.pack(fill="both", expand=True, padx=10, pady=10)

    # Inputs
    _row_browse(frm, "Input folder:", app.input_dir, app.browse_input)
    files_row = tk.Frame(frm); files_row.pack(fill="x", pady=3)
    tk.Button(files_row, text="Pick individual files…", command=app.browse_files).pack(side="left")
    app.files_label = tk.Label(files_row, text="— no files selected —"); app.files_label.pack(side="left", padx=8)
    tk.Button(files_row, text="Clear selection", command=app.clear_files).pack(side="left", padx=8)

    _row_browse(frm, "Output folder (optional):", app.output_dir, app.browse_output)
    tk.Label(frm, text="Results are saved under a 'results/' subfolder. If you don't pick an output folder, "
                       "we use '<input>/results/'.").pack(anchor="w")

    # Weights + engine/device
    _row_browse(frm, "Weights (.pt/.onnx):", app.weights_path, app.browse_weights, is_dir=False)
    row = tk.Frame(frm); row.pack(fill="x", pady=(0,6))
    tk.Label(row, text="Engine:", width=12, anchor="w").pack(side="left")
    for val, txt in [("auto","Auto"),("pt","PyTorch"),("onnx","ONNX")]:
        tk.Radiobutton(row, text=txt, value=val, variable=app.engine_var).pack(side="left", padx=(0,10))
    tk.Label(row, text="Device:", anchor="w").pack(side="left", padx=(12,4))
    tk.Entry(row, textvariable=app.device_var, width=10).pack(side="left")

    # Quality
    qf = tk.LabelFrame(frm, text="Quality (1 = faster/worse, 5 = ULTRA)")
    qf.pack(fill="x", pady=6)
    sc = tk.Scale(qf, from_=1, to=5, orient="horizontal", variable=app.quality, showvalue=True,
                  command=lambda _=None: app._update_preset_label())
    sc.pack(side="left", fill="x", expand=True, padx=6)
    app.preset_label = tk.Label(qf, text=""); app.preset_label.pack(side="left", padx=6)
    app._update_preset_label()

    # AOI
    a = tk.LabelFrame(frm, text="AOI (Area of Interest)")
    a.pack(fill="x", pady=6)
    tk.Checkbutton(a, text="Use AOI (Enter = full image; draw polygons)", variable=app.use_aoi,
                   command=app._on_toggle_use_aoi).pack(anchor="w")
    tk.Checkbutton(a, text="Require AOI for every image before run",
                   variable=app.require_aoi_all).pack(anchor="w")
    tk.Checkbutton(a, text="Persist AOIs to INPUT/aoi & aoi_masks",
                   variable=app.persist_aoi).pack(anchor="w")
    tk.Button(a, text="Open AOI editor…", command=app._open_aoi_editor).pack(anchor="w", pady=(4,0))

    ar = tk.Frame(a); ar.pack(fill="x", pady=(4,0))
    tk.Label(ar, text="AOI mode:", width=14, anchor="w").pack(side="left")
    tk.Radiobutton(ar, text="Centroid inside", variable=app.aoi_mode, value="centroid").pack(side="left", padx=(0,10))
    tk.Radiobutton(ar, text="Box overlap ≥ fraction", variable=app.aoi_mode, value="box").pack(side="left", padx=(0,10))
    tk.Label(ar, text="Min fraction:", anchor="w").pack(side="left", padx=(12,4))
    tk.Entry(ar, textvariable=app.aoi_box_frac, width=6).pack(side="left")

    # Visualization
    vis = tk.LabelFrame(frm, text="Visualization (annotation)")
    vis.pack(fill="x", pady=6)
    tk.Checkbutton(vis, text="Export annotated images", variable=app.annotate).pack(side="left", padx=(0,12))
    tk.Checkbutton(vis, text="Draw centroid dot", variable=app.draw_centroid).pack(side="left", padx=(0,12))
    tk.Label(vis, text="Overlay:", anchor="w").pack(side="left", padx=(12,4))
    ttk.Combobox(vis, values=["boxes_conf","boxes","centroid"], textvariable=app.overlay_mode, state="readonly", width=14
                ).pack(side="left")

    # Classes
    cf = tk.LabelFrame(frm, text="Classes")
    cf.pack(fill="both", expand=True, pady=6)
    tk.Checkbutton(cf, text="Show class checkboxes (filter classes)", variable=app.show_class_checkboxes,
                   command=lambda: _toggle_classes(app)).pack(anchor="w", padx=4, pady=(2,2))
    tk.Checkbutton(cf, text="Require at least one class selected", variable=app.require_class_selection).pack(anchor="w", padx=4)
    # Shorter scroll area + we render checkboxes in 4 columns
    app.class_scroll = app.ScrollableFrame(cf, height=120); app.class_scroll.pack(fill="both", expand=True, padx=2, pady=2)
    app.classes_container = app.class_scroll.inner

    # Actions
    act = tk.Frame(frm); act.pack(fill="x", pady=8)
    app.btn_start = tk.Button(act, text="START", command=app.start); app.btn_start.pack(side="left")
    tk.Button(act, text="Advanced options…", command=app.open_advanced).pack(side="left", padx=8)
    app.btn_abort = tk.Button(act, text="ABORT", command=app.abort, state="disabled"); app.btn_abort.pack(side="left", padx=10)

    # Log + progress
    app.log = tk.Text(frm, height=12); app.log.pack(fill="both", expand=True, pady=(6,2))
    pf = tk.Frame(frm); pf.pack(fill="x", pady=4)
    app.progressbar = ttk.Progressbar(pf, maximum=100.0, variable=app.progress_var); app.progressbar.pack(fill="x")
    tk.Label(pf, textvariable=app.progress_label, anchor="w").pack(fill="x")

def _row_browse(parent, label, var, cmd, is_dir=True):
    f = tk.Frame(parent); f.pack(fill="x", pady=3)
    tk.Label(f, text=label, width=24, anchor="w").pack(side="left")
    tk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True, padx=6)
    tk.Button(f, text="Browse…", command=cmd).pack(side="left")

def _toggle_classes(app):
    for w in app.classes_container.winfo_children():
        w.destroy()
    if not app.show_class_checkboxes.get():
        tk.Label(app.classes_container, text="(Class filtering disabled)").grid(row=0, column=0, sticky="w")
    else:
        app._build_class_checkboxes()

def open_advanced(app):
    win = tk.Toplevel(app); win.title("Advanced options"); win.geometry("660x600")

    tk.Label(win, text="Current preset (from the slider):").pack(anchor="w", padx=8, pady=(8,2))
    from start_app import QUALITY_PRESETS, DEFAULT_QUALITY, auto_wbf_iou, DEFAULT_WBF_ALPHA, \
                          DEFAULT_SEAM_IOU_LOW, DEFAULT_SEAM_BAND_FACTOR, DEFAULT_SEAM_WEIGHT, DEFAULT_MARGIN_WEIGHT
    p = QUALITY_PRESETS.get(int(app.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY])
    preset_txt = tk.StringVar(value=f"tile={p['tile']}  overlap={p['overlap']}  conf={p['conf']}  nms={p['iou_nms']}  WBF={p['use_wbf']}")
    tk.Entry(win, textvariable=preset_txt, state="readonly").pack(fill="x", padx=8)

    current_auto_wbf = auto_wbf_iou(int(app.quality.get()), float(p["iou_nms"]))

    tk.Label(win, text="Override (leave blank = use preset/auto):").pack(anchor="w", padx=8, pady=(10,2))
    def row(lbltxt, var):
        f = tk.Frame(win); f.pack(fill="x", pady=3)
        tk.Label(f, text=lbltxt, width=30, anchor="w").pack(side="left")
        entry = tk.Entry(f, textvariable=var, width=18); entry.pack(side="left"); return entry

    base = app.advanced_params if app.advanced_override else {
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

    row("Tile size (px)", var_tile)
    row("Overlap (0..1)", var_ov)
    row("Confidence", var_conf)
    row("IoU NMS", var_nms)

    f2 = tk.Frame(win); f2.pack(fill="x", pady=4)
    tk.Checkbutton(f2, text="Use WBF deduplication", variable=var_wbf).pack(anchor="w", padx=8)

    f3 = tk.Frame(win); f3.pack(fill="x", pady=3)
    tk.Label(f3, text="WBF alpha", width=30, anchor="w").pack(side="left", padx=8)
    tk.Entry(f3, textvariable=var_alpha, width=18).pack(side="left")

    f4 = tk.Frame(win); f4.pack(fill="x", pady=3)
    cb = tk.Checkbutton(f4, text="Auto WBF IoU (ULTRA=0.60, otherwise=max(0.55, NMS))",
                        variable=var_wbf_auto, command=lambda: toggle_wbf_iou_state())
    cb.pack(anchor="w", padx=8)
    tk.Label(f4, text="WBF IoU (when Auto OFF):", width=30, anchor="w").pack(side="left", padx=8)
    ent_wbf = tk.Entry(f4, textvariable=var_wbf_iou, width=18); ent_wbf.pack(side="left")

    row("Seam-dedup: low IoU", var_seam_iou_low)
    row("Seam-band factor (× step)", var_seam_band_fact)
    row("Weight: margin", var_margin_weight)
    row("Weight: seam distance", var_seam_weight)

    def toggle_wbf_iou_state():
        if var_wbf_auto.get():
            var_wbf_iou.set(str(current_auto_wbf)); ent_wbf.config(state="disabled")
        else:
            var_wbf_iou.set(""); ent_wbf.config(state="normal")
    toggle_wbf_iou_state()

    btns = tk.Frame(win); btns.pack(fill="x", pady=12)
    def apply_override():
        try:
            cur = QUALITY_PRESETS.get(int(app.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY])
            def get_or(v, cast, key):
                s = v.get().strip(); return cast(s) if s!="" else cast(cur[key])
            app.advanced_params = {
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
            app.advanced_override = True
            app._log(f"[ADV] Override enabled. Auto WBF IoU = {app.advanced_params['wbf_auto']}.")
            win.destroy()
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Advanced", str(e))

    def reset_to_preset():
        app.advanced_override = False; app._log("[ADV] Preset restored from the slider."); win.destroy()

    def apply_anti_seam():
        cur = QUALITY_PRESETS.get(int(app.quality.get()), QUALITY_PRESETS[DEFAULT_QUALITY])
        app.advanced_params = {
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
        app.advanced_override = True
        app._log("[ADV] Anti-seam preset applied.")
        win.destroy()

    tk.Button(btns, text="Apply override", command=apply_override).pack(side="left", padx=6)
    tk.Button(btns, text="Restore preset", command=reset_to_preset).pack(side="left", padx=6)
    tk.Button(btns, text="Anti-seam (dedup)", command=apply_anti_seam).pack(side="left", padx=6)