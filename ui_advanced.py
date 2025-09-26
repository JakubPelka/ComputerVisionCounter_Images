# ui_advanced.py — Advanced settings dialog (compact layout, strong dedup macro, full preset fill)
from __future__ import annotations
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Keys used by App.advanced_params (start_app expects these)
_PARAM_KEYS = [
    "tile", "overlap", "conf", "iou_nms",
    "use_wbf", "wbf_auto", "wbf_iou", "wbf_alpha",
    "seam_iou_low", "seam_band_factor", "seam_weight",
    "margin_weight",
]

# Built-in quality presets
_BUILTIN_PRESETS = {
    "Fast":     {"tile": 896,  "overlap": 0.25, "conf": 0.45, "iou_nms": 0.60},
    "Balanced": {"tile": 1024, "overlap": 0.30, "conf": 0.50, "iou_nms": 0.55},
    "Ultra":    {"tile": 2560, "overlap": 0.60, "conf": 0.75, "iou_nms": 0.40},
}
# Defaults paired with any preset
_DEFAULT_FUSION = {"use_wbf": True, "wbf_alpha": 0.20}
_DEFAULT_SEAM   = {"seam_iou_low": 0.30, "seam_band_factor": 0.10, "seam_weight": 0.35, "margin_weight": 0.25}

# A stronger, visible de-dup macro so you can see the change in the UI
_ANTI_SEAM_PRESET = {
    "use_wbf": True,
    "wbf_alpha": 0.25,      # more blending
    "wbf_auto": True,       # keep Auto (leave IoU empty)
    "seam_iou_low": 0.35,   # more strict seam conflict
    "seam_band_factor": 0.15,
    "seam_weight": 0.50,
    "margin_weight": 0.35,
}

# Beginner-friendly help
_HELP = {
    "preset": (
        "Presets fill all fields with sensible starting values.\n\n"
        "• Fast — small tiles and low overlap. Quickest; may miss tiny/edge objects.\n"
        "• Balanced — good everyday trade-off; recommended first try.\n"
        "• Ultra — large tiles and high overlap. Slowest; best quality and fewest seam issues."
    ),
    "tile": (
        "Tile size (imgsz) is the pixel size given to the model per inference.\n"
        "Bigger tiles capture more detail and reduce tiling artifacts, but cost more VRAM/RAM and time.\n"
        "If you hit memory errors, reduce this value."
    ),
    "overlap": (
        "Tile overlap is a fraction (0–1) of how much neighbouring tiles overlap.\n"
        "More overlap smooths seams and improves counts for objects crossing tile boundaries, at a speed cost."
    ),
    "conf": (
        "Confidence threshold (0–1). Detections below this score are discarded.\n"
        "Raise it to reduce false positives, lower it to catch faint/occluded objects."
    ),
    "iou_nms": (
        "NMS IoU (0–1). When two boxes overlap above this value, NMS keeps one.\n"
        "Lower = more aggressive merging (fewer boxes). Higher = keep more boxes."
    ),
    "use_wbf": (
        "Weighted Boxes Fusion (WBF) merges overlapping boxes (especially from different tiles)\n"
        "into a single, better box instead of dropping one like NMS. Keep ON for tiled inference."
    ),
    "wbf_iou": (
        "WBF IoU decides when two boxes are the same object for fusion.\n"
        "Leave EMPTY to auto-select a good value from your quality level.\n"
        "Enter e.g. 0.60 to force a manual value (Auto off)."
    ),
    "wbf_alpha": (
        "WBF alpha sets how strongly confidence influences the fused box.\n"
        "0.20 is safe; slightly higher (0.25–0.35) can stabilise results on difficult scenes."
    ),
    "seam_iou_low": (
        "Low IoU with seam (0–1). If boxes across a tile border overlap less than this,\n"
        "treat them as seam conflicts and resolve with voting to avoid double-counts."
    ),
    "seam_band_factor": (
        "Relative width of the special seam band along tile edges. Larger widens the zone\n"
        "where anti-seam voting applies. Helps when many objects cross tile borders."
    ),
    "seam_weight": (
        "Voting weight for seam-band boxes. Higher gives seam candidates more say\n"
        "when deciding which overlapping boxes to keep."
    ),
    "margin_weight": (
        "Voting weight near image margins. Helpful if objects often sit at the very edge."
    ),
}

def _get(app, key, default=None):
    try:    return app.advanced_params.get(key, default)
    except: return default

def _float(v, default=0.0):
    try:    return float(v)
    except: return default

def _int(v, default=0):
    try:    return int(float(v))
    except: return default

def _apply_to_app(app, values: dict, log_msg: str | None = None):
    for k in _PARAM_KEYS:
        if k in values:
            app.advanced_params[k] = values[k]
    app.advanced_override = True
    try: app._update_preset_label()
    except: pass
    if log_msg:
        try: app._log(log_msg)
        except: pass


def open_advanced(app):
    # Reuse existing dialog
    existing = getattr(app, "_advanced_win", None)
    if existing and existing.winfo_exists():
        existing.deiconify(); existing.lift(); existing.focus_force()
        return

    win = tk.Toplevel(app)
    app._advanced_win = win
    win.title("Advanced options")
    win.transient(app)

    # ----- bind to app params
    var_tile   = tk.StringVar(value=str(_get(app, "tile", 1024)))
    var_ovl    = tk.StringVar(value=str(_get(app, "overlap", 0.30)))
    var_conf   = tk.StringVar(value=str(_get(app, "conf", 0.50)))
    var_iou    = tk.StringVar(value=str(_get(app, "iou_nms", 0.55)))

    var_use_wbf = tk.BooleanVar(value=bool(_get(app, "use_wbf", True)))
    init_wbf_iou = _get(app, "wbf_iou", None)
    init_auto = bool(_get(app, "wbf_auto", True))
    var_wbf_iou = tk.StringVar(value="" if (init_auto or init_wbf_iou in (None, "")) else str(init_wbf_iou))
    var_wbf_alp = tk.StringVar(value=str(_get(app, "wbf_alpha", 0.20)))

    var_seam_iou_low  = tk.StringVar(value=str(_get(app, "seam_iou_low", 0.30)))
    var_seam_band     = tk.StringVar(value=str(_get(app, "seam_band_factor", 0.10)))
    var_seam_weight   = tk.StringVar(value=str(_get(app, "seam_weight", 0.35)))
    var_margin_weight = tk.StringVar(value=str(_get(app, "margin_weight", 0.25)))

    # ===== LAYOUT: main area (left form + right help) + bottom action bar
    main = tk.Frame(win); main.pack(side="top", fill="both", expand=True, padx=8, pady=(8,4))

    # Left column: do NOT expand — keeps it tight and eliminates empty space
    left = tk.Frame(main)
    left.pack(side="left", anchor="n", fill="y", expand=False)

    # Right Help panel: fixed width, sticks to top
    right = tk.Frame(main, width=300)
    right.pack(side="left", anchor="n", fill="y", padx=(10,0), expand=False)
    right.pack_propagate(False)

    help_title = tk.Label(right, text="Help", font=("TkDefaultFont", 10, "bold"))
    help_title.pack(anchor="w")
    help_text = tk.Label(right, text=_HELP["preset"], wraplength=280, justify="left", fg="#444")
    help_text.pack(anchor="n", pady=(4,0))
    def set_help(key: str): help_text.config(text=_HELP.get(key, ""))

    # --- Presets + import/export (compact)
    pr = tk.LabelFrame(left, text="Presets"); pr.pack(anchor="n", fill="x", pady=(0,4))
    tk.Label(pr, text="Preset:").pack(side="left", padx=(6,2))
    var_preset = tk.StringVar(value="Custom")
    cb = ttk.Combobox(pr, values=["Fast","Balanced","Ultra","Custom"], textvariable=var_preset, state="readonly", width=12)
    cb.pack(side="left")
    cb.bind("<<ComboboxSelected>>", lambda _e: _apply_builtin(var_preset.get()))
    _add_help_bindings(cb, lambda: set_help("preset"))
    tk.Button(pr, text="Load preset…", command=lambda: _import_preset(win, app,
            _binds_to_dict(var_tile,var_ovl,var_conf,var_iou,var_use_wbf,var_wbf_iou,var_wbf_alp,
                           var_seam_iou_low,var_seam_band,var_seam_weight,var_margin_weight))
             ).pack(side="left", padx=(10,0))
    tk.Button(pr, text="Save preset as…", command=lambda: _export_preset(win, app,
            _binds_to_dict(var_tile,var_ovl,var_conf,var_iou,var_use_wbf,var_wbf_iou,var_wbf_alp,
                           var_seam_iou_low,var_seam_band,var_seam_weight,var_margin_weight))
             ).pack(side="left", padx=6)

    # --- Inference / tiling
    g1 = tk.LabelFrame(left, text="Inference & Tiling"); g1.pack(anchor="n", fill="x", pady=3)
    _row(g1, "Tile size (imgsz):", var_tile, lambda: set_help("tile"))
    _row(g1, "Tile overlap (0–1):", var_ovl, lambda: set_help("overlap"))
    _row(g1, "Conf threshold (0–1):", var_conf, lambda: set_help("conf"))
    _row(g1, "NMS IoU (0–1):", var_iou, lambda: set_help("iou_nms"))

    # --- WBF (stacked)
    g2 = tk.LabelFrame(left, text="Weighted Boxes Fusion (WBF)"); g2.pack(anchor="n", fill="x", pady=3)
    cb_use = tk.Checkbutton(g2, text="Use WBF", variable=var_use_wbf, command=lambda: set_help("use_wbf"))
    cb_use.pack(anchor="w", padx=6, pady=(3,1))
    cb_use.bind("<FocusIn>", lambda _e: set_help("use_wbf"))
    _row(g2, "WBF IoU (empty = Auto):", var_wbf_iou, lambda: set_help("wbf_iou"))
    _row(g2, "WBF alpha:", var_wbf_alp, lambda: set_help("wbf_alpha"))

    # --- Seam / de-dup
    g3 = tk.LabelFrame(left, text="Seam/Border de-duplication"); g3.pack(anchor="n", fill="x", pady=3)
    _row(g3, "Low IoU with seam (0–1):", var_seam_iou_low, lambda: set_help("seam_iou_low"))
    _row(g3, "Seam band factor (0–1):", var_seam_band, lambda: set_help("seam_band_factor"))
    _row(g3, "Seam vote weight (0–1):", var_seam_weight, lambda: set_help("seam_weight"))
    _row(g3, "Margin vote weight (0–1):", var_margin_weight, lambda: set_help("margin_weight"))

    # ===== Bottom bar — tight, no extra vertical space
    bar = tk.Frame(win); bar.pack(side="bottom", fill="x", padx=8, pady=(0,8))
    status = tk.Label(bar, text="", fg="#2a7")
    status.pack(side="left")

    def apply_and_stay(msg=None):
        vals = _binds_to_dict(var_tile,var_ovl,var_conf,var_iou,var_use_wbf,var_wbf_iou,var_wbf_alp,
                              var_seam_iou_low,var_seam_band,var_seam_weight,var_margin_weight)
        # validate/coerce
        vals["tile"]  = _int(vals["tile"], 1024)
        vals["overlap"] = max(0.0, min(1.0, _float(vals["overlap"], 0.3)))
        vals["conf"]  = max(0.0, min(1.0, _float(vals["conf"], 0.5)))
        vals["iou_nms"] = max(0.0, min(1.0, _float(vals["iou_nms"], 0.55)))
        vals["use_wbf"] = bool(vals["use_wbf"])
        vals["wbf_alpha"] = max(0.0, min(1.0, _float(vals["wbf_alpha"], 0.2)))
        txt = vals.get("wbf_iou","")
        if txt in ("", None):
            vals["wbf_iou"] = None; vals["wbf_auto"] = True
        else:
            vals["wbf_iou"] = max(0.0, min(1.0, _float(txt, 0.6))); vals["wbf_auto"] = False
        vals["seam_iou_low"]  = max(0.0, min(1.0, _float(vals["seam_iou_low"], 0.30)))
        vals["seam_band_factor"] = max(0.0, min(1.0, _float(vals["seam_band_factor"], 0.10)))
        vals["seam_weight"]   = max(0.0, min(1.0, _float(vals["seam_weight"], 0.35)))
        vals["margin_weight"] = max(0.0, min(1.0, _float(vals["margin_weight"], 0.25)))
        _apply_to_app(app, vals, log_msg=msg or "[ADV] Parameters applied.")
        status.config(text="Applied.", fg="#2a7")

    def ok_and_close():
        apply_and_stay("[ADV] Parameters applied.")
        try: win.destroy()
        except: pass

    def cancel():
        try: win.destroy()
        except: pass

    def apply_anti_seam():
        # Reflect macro into visible fields so you SEE the change
        var_use_wbf.set(bool(_ANTI_SEAM_PRESET.get("use_wbf", True)))
        if _ANTI_SEAM_PRESET.get("wbf_auto", True):
            var_wbf_iou.set("")  # Auto
        else:
            iou = _ANTI_SEAM_PRESET.get("wbf_iou", "")
            var_wbf_iou.set("" if iou in (None, "") else str(iou))
        var_wbf_alp.set(str(_ANTI_SEAM_PRESET.get("wbf_alpha", 0.20)))
        var_seam_iou_low.set(str(_ANTI_SEAM_PRESET.get("seam_iou_low", 0.30)))
        var_seam_band.set(str(_ANTI_SEAM_PRESET.get("seam_band_factor", 0.10)))
        var_seam_weight.set(str(_ANTI_SEAM_PRESET.get("seam_weight", 0.35)))
        var_margin_weight.set(str(_ANTI_SEAM_PRESET.get("margin_weight", 0.25)))
        apply_and_stay("[ADV] Anti-seam preset applied.")

    tk.Button(bar, text="Apply anti-seam dedup", command=apply_anti_seam).pack(side="left")
    tk.Button(bar, text="Cancel", command=cancel).pack(side="right")
    tk.Button(bar, text="OK", command=ok_and_close).pack(side="right", padx=6)
    tk.Button(bar, text="Apply", command=apply_and_stay).pack(side="right")

    # keep dialog above main initially
    win.lift()
    try:
        win.attributes("-topmost", True)
        win.after(250, lambda: win.attributes("-topmost", False))
    except: pass

    # --- internal: apply preset (fills ALL sections) ---
    def _apply_builtin(name: str):
        base = _BUILTIN_PRESETS.get(name)
        if not base:
            set_help("preset"); return
        var_tile.set(str(base["tile"]))
        var_ovl.set(str(base["overlap"]))
        var_conf.set(str(base["conf"]))
        var_iou.set(str(base["iou_nms"]))
        # Pair with defaults so WBF & Seam also change on preset switch
        var_use_wbf.set(bool(_DEFAULT_FUSION["use_wbf"]))
        var_wbf_iou.set("")  # Auto
        var_wbf_alp.set(str(_DEFAULT_FUSION["wbf_alpha"]))
        var_seam_iou_low.set(str(_DEFAULT_SEAM["seam_iou_low"]))
        var_seam_band.set(str(_DEFAULT_SEAM["seam_band_factor"]))
        var_seam_weight.set(str(_DEFAULT_SEAM["seam_weight"]))
        var_margin_weight.set(str(_DEFAULT_SEAM["margin_weight"]))
        apply_and_stay(f"[ADV] Built-in preset applied: {name}")

    # ---------- Auto-size to content (removes right/bottom empty space) ----------
    win.update_idletasks()
    req_w = left.winfo_reqwidth() + right.winfo_reqwidth() + 40
    req_h = max(left.winfo_reqheight(), right.winfo_reqheight()) + bar.winfo_reqheight() + 40
    req_w = max(760, req_w)   # keep a sensible min
    req_h = max(520, req_h)
    try:
        win.geometry(f"{req_w}x{req_h}")
        win.minsize(req_w, req_h)
    except Exception:
        pass


# ---------- helpers ----------
def _row(parent, label, var: tk.StringVar, focus_cb=None):
    f = tk.Frame(parent); f.pack(fill="x", padx=6, pady=1)
    tk.Label(f, text=label, width=26, anchor="w").pack(side="left")
    e = tk.Entry(f, textvariable=var, width=12); e.pack(side="left")
    if focus_cb: e.bind("<FocusIn>", lambda _e: focus_cb())

def _binds_to_dict(*vars_) -> dict:
    (var_tile,var_ovl,var_conf,var_iou,var_use_wbf,var_wbf_iou,var_wbf_alp,
     var_seam_iou_low,var_seam_band,var_seam_weight,var_margin_weight) = vars_
    return {
        "tile": var_tile.get(),
        "overlap": var_ovl.get(),
        "conf": var_conf.get(),
        "iou_nms": var_iou.get(),
        "use_wbf": var_use_wbf.get(),
        "wbf_iou": var_wbf_iou.get(),
        "wbf_alpha": var_wbf_alp.get(),
        "seam_iou_low": var_seam_iou_low.get(),
        "seam_band_factor": var_seam_band.get(),
        "seam_weight": var_seam_weight.get(),
        "margin_weight": var_margin_weight.get(),
    }

def _add_help_bindings(widget, on_focus):
    try: widget.bind("<FocusIn>", lambda _e: on_focus())
    except: pass

def _import_preset(win, app, cur_vals: dict):
    fp = filedialog.askopenfilename(
        title="Load advanced preset",
        filetypes=[("JSON files","*.json"), ("All files","*.*")],
        parent=win
    )
    if not fp: return
    try:
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Preset JSON must be an object with parameter keys.")
        merged = {k: data[k] for k in _PARAM_KEYS if k in data}
        # Reflect UI (best-effort; if called with raw values, this still applies to app)
        to_apply = {
            "tile": int(float(merged.get("tile", cur_vals["tile"]))),
            "overlap": float(merged.get("overlap", cur_vals["overlap"])),
            "conf": float(merged.get("conf", cur_vals["conf"])),
            "iou_nms": float(merged.get("iou_nms", cur_vals["iou_nms"])),
            "use_wbf": bool(merged.get("use_wbf", cur_vals["use_wbf"])),
            "wbf_alpha": float(merged.get("wbf_alpha", cur_vals["wbf_alpha"])),
            "wbf_iou": None if merged.get("wbf_auto", True) else float(merged.get("wbf_iou", 0.6)),
            "wbf_auto": bool(merged.get("wbf_auto", True)),
            "seam_iou_low": float(merged.get("seam_iou_low", cur_vals["seam_iou_low"])),
            "seam_band_factor": float(merged.get("seam_band_factor", cur_vals["seam_band_factor"])),
            "seam_weight": float(merged.get("seam_weight", cur_vals["seam_weight"])),
            "margin_weight": float(merged.get("margin_weight", cur_vals["margin_weight"])),
        }
        _apply_to_app(app, to_apply, f"[ADV] Preset loaded: {Path(fp).name}")
        messagebox.showinfo("Advanced", f"Preset loaded:\n{fp}", parent=win)
    except Exception as e:
        messagebox.showerror("Advanced", f"Failed to load preset:\n{e}", parent=win)

def _export_preset(win, app, cur_vals: dict):
    # Handle both bool and tk.BooleanVar
    use_wbf_val = cur_vals.get("use_wbf")
    use_wbf = bool(use_wbf_val.get()) if hasattr(use_wbf_val, "get") else bool(use_wbf_val)
    wbf_iou_txt = cur_vals.get("wbf_iou")
    wbf_auto = True if wbf_iou_txt in ("", None) else False

    to_save = {
        "tile": int(float(cur_vals["tile"])),
        "overlap": float(cur_vals["overlap"]),
        "conf": float(cur_vals["conf"]),
        "iou_nms": float(cur_vals["iou_nms"]),
        "use_wbf": use_wbf,
        "wbf_auto": wbf_auto,
        "wbf_iou": None if wbf_auto else float(wbf_iou_txt),
        "wbf_alpha": float(cur_vals["wbf_alpha"]),
        "seam_iou_low": float(cur_vals["seam_iou_low"]),
        "seam_band_factor": float(cur_vals["seam_band_factor"]),
        "seam_weight": float(cur_vals["seam_weight"]),
        "margin_weight": float(cur_vals["margin_weight"]),
    }
    default_dir = Path.cwd() / "presets"; default_dir.mkdir(parents=True, exist_ok=True)
    fp = filedialog.asksaveasfilename(
        title="Save advanced preset", defaultextension=".json",
        initialdir=str(default_dir), initialfile="cv_counter_preset.json",
        filetypes=[("JSON files","*.json")], parent=win
    )
    if not fp: return
    try:
        Path(fp).write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
        messagebox.showinfo("Advanced", f"Preset saved:\n{fp}", parent=win)
    except Exception as e:
        messagebox.showerror("Advanced", f"Failed to save preset:\n{e}", parent=win)
