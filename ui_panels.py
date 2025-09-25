# ui_panels.py — UI builders (English), same layout; Progress frame above log
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from widgets import ProgressCanvas

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
    app.class_scroll = app.ScrollableFrame(cf, height=120); app.class_scroll.pack(fill="both", expand=True, padx=2, pady=2)
    app.classes_container = app.class_scroll.inner
    app.classes_container.bind("<Configure>", lambda e: getattr(app, "_reflow_class_grid", lambda *_:None)(e.width))

    # Actions
    act = tk.Frame(frm); act.pack(fill="x", pady=8)
    app.btn_start = tk.Button(act, text="START", command=app.start); app.btn_start.pack(side="left")
    tk.Button(act, text="Advanced options…", command=app.open_advanced).pack(side="left", padx=8)
    app.btn_abort = tk.Button(act, text="ABORT", command=app.abort, state="disabled"); app.btn_abort.pack(side="left", padx=10)

    # Progress (nad logiem, zawsze widoczny)
    pf = tk.LabelFrame(frm, text="Progress"); pf.pack(fill="x", pady=(4,2))
    app.progressbar = ttk.Progressbar(pf, maximum=100.0, variable=app.progress_var, mode="determinate")
    app.progressbar.pack(fill="x")
    ProgressCanvas(pf, app.progress_var, height=22).pack(fill="x", pady=(4,2))
    tk.Label(pf, textvariable=app.progress_label, anchor="w").pack(fill="x")

    # Log (pod spodem)
    app.log = tk.Text(frm, height=12); app.log.pack(fill="both", expand=True, pady=(6,2))

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
