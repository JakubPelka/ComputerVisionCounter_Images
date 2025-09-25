# ui_panels.py — layout & panels (English), AOI controls + import/export,
# Advanced beside Quality slider, dynamic classes grid hook.
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from widgets import ScrollableFrame

def build_main_ui(app):
    root = tk.Frame(app); root.pack(fill="both", expand=True, padx=10, pady=8)

    # ========== Paths ==========
    paths = tk.LabelFrame(root, text="Paths"); paths.pack(fill="x", pady=(0,6))
    _row_browse(paths, "Input folder (images):", app.input_dir, app.browse_input, is_dir=True)
    files = tk.Frame(paths); files.pack(fill="x", pady=2)
    tk.Button(files, text="Pick images…", command=app.browse_files).pack(side="left")
    app.files_label = tk.Label(files, text="— no files selected —"); app.files_label.pack(side="left", padx=8)
    tk.Button(files, text="Clear", command=app.clear_files).pack(side="left", padx=(8,0))
    _row_browse(paths, "Output folder (optional):", app.output_dir, app.browse_output, is_dir=True)
    _row_browse(paths, "Weights (.pt / .onnx):", app.weights_path, app.browse_weights, is_dir=False)

    # ========== Engine / Device ==========
    ed = tk.LabelFrame(root, text="Engine & Device"); ed.pack(fill="x", pady=(0,6))
    tk.Label(ed, text="Engine:").pack(side="left", padx=(6,2))
    ttk.Combobox(ed, values=["auto","pt","onnx"], textvariable=app.engine_var,
                 width=8, state="readonly").pack(side="left")
    tk.Label(ed, text="Device:").pack(side="left", padx=(12,2))
    ttk.Combobox(ed, values=["auto","cpu","cuda:0"], textvariable=app.device_var,
                 width=10, state="readonly").pack(side="left")

    # ========== Quality ==========
    qf = tk.LabelFrame(root, text="Quality (1 = faster • 5 = ultra)"); qf.pack(fill="x", pady=(0,6))
    left = tk.Frame(qf); left.pack(side="left", fill="x", expand=True)
    tk.Scale(left, from_=1, to=5, orient="horizontal", variable=app.quality,
             command=lambda *_: app._update_preset_label()).pack(side="left", fill="x", expand=True, padx=6, pady=2)
    app.preset_label = tk.Label(left, text=""); app.preset_label.pack(side="left", padx=6)
    app._update_preset_label()
    tk.Button(qf, text="Advanced…", command=app.open_advanced).pack(side="right", padx=6, pady=2)

    # ========== AOI ==========
    aoi = tk.LabelFrame(root, text="Areas of Interest (AOI)"); aoi.pack(fill="x", pady=(0,6))
    tk.Checkbutton(aoi, text="Require AOI for every image", variable=app.require_aoi_all).pack(anchor="w")
    tk.Checkbutton(aoi, text="Use AOI (draw polygons — finish with Ctrl+Enter, undo vertex: Ctrl+Backspace)",
                   variable=app.use_aoi, command=app._on_toggle_use_aoi).pack(anchor="w")

    # AOI mode row
    m = tk.Frame(aoi); m.pack(fill="x", pady=(4,2))
    tk.Label(m, text="Count within AOI by:").pack(side="left")
    tk.Radiobutton(m, text="Centroid inside polygon", variable=app.aoi_mode, value="centroid").pack(side="left", padx=8)
    tk.Radiobutton(m, text="Box-area fraction", variable=app.aoi_mode, value="box").pack(side="left", padx=8)
    tk.Label(m, text="min box fraction inside AOI:").pack(side="left", padx=(12,2))
    tk.Spinbox(m, from_=0.05, to=1.0, increment=0.05, width=5,
               textvariable=app.aoi_box_frac).pack(side="left")

    # AOI actions row
    act = tk.Frame(aoi); act.pack(fill="x", pady=(4,2))
    tk.Button(act, text="Open AOI editor…", command=app._open_aoi_editor).pack(side="left")
    tk.Button(act, text="Import AOIs", command=app.import_aois_from_input).pack(side="left", padx=6)
    tk.Button(act, text="Export AOIs", command=app.export_aois_now).pack(side="left")

    # ========== Visualization ==========
    vis = tk.LabelFrame(root, text="Visualization"); vis.pack(fill="x", pady=(0,6))
    tk.Label(vis, text="Overlay:").pack(side="left")
    tk.Radiobutton(vis, text="Boxes", value="boxes", variable=app.overlay_mode).pack(side="left", padx=6)
    tk.Radiobutton(vis, text="Boxes + conf", value="boxes_conf", variable=app.overlay_mode).pack(side="left", padx=6)
    tk.Radiobutton(vis, text="Centroid dots", value="centroid", variable=app.overlay_mode).pack(side="left", padx=6)
    tk.Checkbutton(vis, text="Draw centroid dot", variable=app.draw_centroid).pack(side="left", padx=(12,0))
    tk.Checkbutton(vis, text="Save annotated images", variable=app.annotate).pack(side="left", padx=(12,0))

    # ========== Classes (shorter height; scrollbar auto-hides) ==========
    cl = tk.LabelFrame(root, text="Classes (load weights to populate)"); cl.pack(fill="both", expand=True, pady=(0,6))
    app.classes_scroll = ScrollableFrame(cl, height=120)  # ~half height
    app.classes_scroll.pack(fill="both", expand=True)
    app.classes_container = app.classes_scroll.inner
    app.classes_scroll.canvas.bind("<Configure>", lambda e: app._on_classes_canvas_config(e.width))

    # ========== Run / Progress ==========
    rf = tk.Frame(root); rf.pack(fill="x", pady=(0,4))
    app.btn_start = tk.Button(rf, text="START", command=app.start); app.btn_start.pack(side="left")
    app.btn_abort = tk.Button(rf, text="ABORT", command=app.abort, state="disabled"); app.btn_abort.pack(side="left", padx=(8,0))

    pf = tk.Frame(root); pf.pack(fill="x", pady=(2,6))
    app.progressbar = ttk.Progressbar(pf, maximum=100.0, variable=app.progress_var, mode="determinate")
    app.progressbar.pack(fill="x", side="left", expand=True)
    tk.Label(pf, textvariable=app.progress_label, width=32, anchor="w").pack(side="left", padx=8)

    # ========== Log ==========
    logf = tk.LabelFrame(root, text="Log"); logf.pack(fill="both", expand=True)
    app.log = tk.Text(logf, height=10); app.log.pack(fill="both", expand=True)

def _row_browse(parent, label, var, cmd, is_dir=True):
    f = tk.Frame(parent); f.pack(fill="x", pady=2)
    tk.Label(f, text=label, width=26, anchor="w").pack(side="left")
    tk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True, padx=6)
    tk.Button(f, text="Browse…", command=cmd).pack(side="left")
