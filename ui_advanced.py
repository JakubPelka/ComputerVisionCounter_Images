# ui_advanced.py — Advanced settings dialog (presets, contextual help, import/export)
from __future__ import annotations
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Keys used by App.advanced_params
_PARAM_KEYS = [
    "tile", "overlap", "conf", "iou_nms",
    "use_wbf", "wbf_auto", "wbf_iou", "wbf_alpha",
    "seam_iou_low", "seam_band_factor", "seam_weight",
    "margin_weight",
]

# Built-in quality presets (aligned with start_app QUALITY_PRESETS w/ friendly names)
_BUILTIN_PRESETS = {
    "Fast":     {"tile": 896,  "overlap": 0.25, "conf": 0.45, "iou_nms": 0.60},
    "Balanced": {"tile": 1024, "overlap": 0.30, "conf": 0.50, "iou_nms": 0.55},
    "Ultra":    {"tile": 2560, "overlap": 0.60, "conf": 0.75, "iou_nms": 0.40},
}
# Defaults for fusion & seam handling that pair well with any preset
_DEFAULT_FUSION = {"use_wbf": True, "wbf_auto": True, "wbf_alpha": 0.20}
_DEFAULT_SEAM   = {"seam_iou_low": 0.30, "seam_band_factor": 0.10, "seam_weight": 0.35, "margin_weight": 0.25}

# Anti-seam / dedup macro (doesn't close the dialog)
_ANTI_SEAM_PRESET = {
    "use_wbf": True,
    "wbf_alpha": 0.20,
    "wbf_auto": True,   # let app compute default if user doesn’t set wbf_iou
    # "wbf_iou": 0.60,  # set this AND wbf_auto=False if you want fixed value
    "seam_iou_low": 0.30,
    "seam_band_factor": 0.10,
    "seam_weight": 0.35,
    "margin_weight": 0.25,
}

# Context help for the info panel on the right
_HELP = {
    "tile": "Image size (pixels) fed into the model. Larger → better quality but slower/more RAM/VRAM.",
    "overlap": "Tile overlap fraction (0–1). Higher reduces seam artifacts; try 0.45+ for high quality.",
    "conf": "Minimum confidence (0–1) for a detection to be kept. Higher = fewer false positives.",
    "iou_nms": "IoU threshold for Non-Max Suppression. Lower merges more overlaps; 0.50–0.60 is common.",
    "use_wbf": "Weighted Boxes Fusion combines overlapping boxes from tiles for cleaner results.",
    "wbf_auto": "Let the app pick a sensible WBF IoU from quality level; turn off to set your own.",
    "wbf_iou": "Matching IoU used by WBF when Auto is off. 0.55–0.65 usually works well.",
    "wbf_alpha": "How strongly WBF blends boxes. 0.20 is a safe default.",
    "seam_iou_low": "If two boxes across a tile border overlap less than this, treat as seam conflict.",
    "seam_band_factor": "Relative width of the seam band along tile edges used for deduplication.",
    "seam_weight": "Voting weight for boxes affected by seams/borders.",
    "margin_weight": "Voting weight for boxes near the image margins.",
    "preset": "Select a built-in starting point. You can still tweak values afterwards.",
}

def _get(app, key, default=None):
    try:
        return app.advanced_params.get(key, default)
    except Exception:
        return default

def _float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

def _int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def _apply_to_app(app, values: dict, log_msg: str | None = None):
    for k in _PARAM_KEYS:
        if k in values:
            app.advanced_params[k] = values[k]
    app.advanced_override = True
    try:
        app._update_preset_label()
    except Exception:
        pass
    if log_msg:
        try: app._log(log_msg)
        except Exception: pass


def open_advanced(app):
    """Advanced dialog; multiple opens reuse one window if present."""
    existing = getattr(app, "_advanced_win", None)
    if existing and existing.winfo_exists():
        existing.deiconify(); existing.lift(); existing.focus_force()
        return

    win = tk.Toplevel(app)
    app._advanced_win = win
    win.title("Advanced options")
    win.transient(app)
    win.geometry("860x640")          # bigger so buttons stay visible
    try:
        win.minsize(800, 600)
    except Exception:
        pass

    # --- form variables bound to app.advanced_params
    var_tile   = tk.StringVar(value=str(_get(app, "tile", 1024)))
    var_ovl    = tk.StringVar(value=str(_get(app, "overlap", 0.30)))
    var_conf   = tk.StringVar(value=str(_get(app, "conf", 0.50)))
    var_iou    = tk.StringVar(value=str(_get(app, "iou_nms", 0.55)))

    var_use_wbf = tk.BooleanVar(value=bool(_get(app, "use_wbf", True)))
    var_wbf_auto= tk.BooleanVar(value=bool(_get(app, "wbf_auto", True)))
    var_wbf_iou = tk.StringVar(value="" if _get(app, "wbf_iou", None) in (None, "") else str(_get(app,"wbf_iou")))
    var_wbf_alp = tk.StringVar(value=str(_get(app, "wbf_alpha", 0.20)))

    var_seam_iou_low  = tk.StringVar(value=str(_get(app, "seam_iou_low", 0.30)))
    var_seam_band     = tk.StringVar(value=str(_get(app, "seam_band_factor", 0.10)))
    var_seam_weight   = tk.StringVar(value=str(_get(app, "seam_weight", 0.35)))
    var_margin_weight = tk.StringVar(value=str(_get(app, "margin_weight", 0.25)))

    # -------- layout: left form + right help panel
    root = tk.Frame(win); root.pack(fill="both", expand=True, padx=10, pady=10)
    left = tk.Frame(root); left.pack(side="left", fill="both", expand=True)
    right = tk.Frame(root, width=260); right.pack(side="left", fill="y", padx=(12,0))
    right.pack_propagate(False)
    help_title = tk.Label(right, text="Help", font=("TkDefaultFont", 10, "bold"))
    help_title.pack(anchor="w")
    help_text = tk.Label(right, text=_HELP["preset"], wraplength=240, justify="left", fg="#444")
    help_text.pack(fill="x", pady=(4,0))

    def set_help(key: str):
        help_text.config(text=_HELP.get(key, ""))

    # ---- Preset row (with import/export)
    pr = tk.LabelFrame(left, text="Presets"); pr.pack(fill="x", pady=(0,8))
    tk.Label(pr, text="Preset:").pack(side="left", padx=(6,2))
    var_preset = tk.StringVar(value="Custom")
    cb = ttk.Combobox(pr, values=["Fast","Balanced","Ultra","Custom"], textvariable=var_preset, state="readonly", width=12)
    cb.pack(side="left")
    cb.bind("<<ComboboxSelected>>", lambda _e: _apply_builtin(var_preset.get()))
    _add_help_bindings(cb, lambda: set_help("preset"))

    tk.Button(pr, text="Load preset…", command=lambda: _import_preset(win, app,
            _binds_to_dict(var_tile,var_ovl,var_conf,var_iou,var_use_wbf,var_wbf_auto,var_wbf_iou,var_wbf_alp,
                           var_seam_iou_low,var_seam_band,var_seam_weight,var_margin_weight))
             ).pack(side="left", padx=(10,0))
    tk.Button(pr, text="Save preset as…", command=lambda: _export_preset(win, app,
            _binds_to_dict(var_tile,var_ovl,var_conf,var_iou,var_use_wbf,var_wbf_auto,var_wbf_iou,var_wbf_alp,
                           var_seam_iou_low,var_seam_band,var_seam_weight,var_margin_weight))
             ).pack(side="left", padx=6)

    # ---- Inference / tiling group
    g1 = tk.LabelFrame(left, text="Inference & Tiling"); g1.pack(fill="x", pady=6)
    _row(g1, "Tile size (imgsz):", var_tile, lambda: set_help("tile"))
    _row(g1, "Tile overlap (0–1):", var_ovl, lambda: set_help("overlap"))
    _row(g1, "Conf threshold (0–1):", var_conf, lambda: set_help("conf"))
    _row(g1, "NMS IoU (0–1):", var_iou, lambda: set_help("iou_nms"))

    # ---- WBF group (stacked checkboxes)
    g2 = tk.LabelFrame(left, text="Weighted Boxes Fusion (WBF)"); g2.pack(fill="x", pady=6)
    cb_use = tk.Checkbutton(g2, text="Use WBF", variable=var_use_wbf, command=lambda: set_help("use_wbf"))
    cb_use.pack(anchor="w", padx=6, pady=(4,2))
    cb_use.bind("<FocusIn>", lambda _e: set_help("use_wbf"))

    cb_auto = tk.Checkbutton(g2, text="Auto WBF IoU", variable=var_wbf_auto, command=lambda: set_help("wbf_auto"))
    cb_auto.pack(anchor="w", padx=6, pady=(0,4))
    cb_auto.bind("<FocusIn>", lambda _e: set_help("wbf_auto"))

    _row(g2, "WBF IoU (if not auto):", var_wbf_iou, lambda: set_help("wbf_iou"))
    _row(g2, "WBF alpha:", var_wbf_alp, lambda: set_help("wbf_alpha"))

    # ---- Seam / dedup group
    g3 = tk.LabelFrame(left, text="Seam/Border de-duplication"); g3.pack(fill="x", pady=6)
    _row(g3, "Low IoU with seam (0–1):", var_seam_iou_low, lambda: set_help("seam_iou_low"))
    _row(g3, "Seam band factor (0–1):", var_seam_band, lambda: set_help("seam_band_factor"))
    _row(g3, "Seam vote weight (0–1):", var_seam_weight, lambda: set_help("seam_weight"))
    _row(g3, "Margin vote weight (0–1):", var_margin_weight, lambda: set_help("margin_weight"))

    # ---- Actions row
    ar = tk.Frame(left); ar.pack(fill="x", pady=(10,0))
    status = tk.Label(ar, text="", fg="#2a7"); status.pack(side="left")

    def apply_and_stay(msg=None):
        vals = _binds_to_dict(var_tile,var_ovl,var_conf,var_iou,var_use_wbf,var_wbf_auto,var_wbf_iou,var_wbf_alp,
                              var_seam_iou_low,var_seam_band,var_seam_weight,var_margin_weight)
        # validation
        vals["tile"] = _int(vals["tile"], 1024)
        vals["overlap"] = max(0.0, min(1.0, _float(vals["overlap"], 0.3)))
        vals["conf"] = max(0.0, min(1.0, _float(vals["conf"], 0.5)))
        vals["iou_nms"] = max(0.0, min(1.0, _float(vals["iou_nms"], 0.55)))
        vals["use_wbf"] = bool(vals["use_wbf"])
        vals["wbf_auto"] = bool(vals["wbf_auto"])
        vals["wbf_alpha"] = max(0.0, min(1.0, _float(vals["wbf_alpha"], 0.2)))
        wbf_iou_txt = vals.get("wbf_iou", "")
        vals["wbf_iou"] = None if (wbf_iou_txt in ("", None) or vals["wbf_auto"]) else max(0.0, min(1.0, _float(wbf_iou_txt, 0.6)))
        vals["seam_iou_low"] = max(0.0, min(1.0, _float(vals["seam_iou_low"], 0.30)))
        vals["seam_band_factor"] = max(0.0, min(1.0, _float(vals["seam_band_factor"], 0.10)))
        vals["seam_weight"] = max(0.0, min(1.0, _float(vals["seam_weight"], 0.35)))
        vals["margin_weight"] = max(0.0, min(1.0, _float(vals["margin_weight"], 0.25)))
        _apply_to_app(app, vals, log_msg=msg or "[ADV] Parameters applied.")
        status.config(text="Applied.", fg="#2a7")

    def ok_and_close():
        apply_and_stay(msg="[ADV] Parameters applied.")
        try: win.destroy()
        except Exception: pass

    def cancel():
        try: win.destroy()
        except Exception: pass

    def apply_anti_seam():
        # Overlay the macro and reflect it in UI (keep dialog open)
        for k, v in _ANTI_SEAM_PRESET.items():
            if k == "wbf_iou" and v is None:
                continue
            if k == "use_wbf": var_use_wbf.set(bool(v))
            elif k == "wbf_auto": var_wbf_auto.set(bool(v))
            elif k == "wbf_iou": var_wbf_iou.set("" if v is None else str(v))
            elif k == "wbf_alpha": var_wbf_alp.set(str(v))
            elif k == "seam_iou_low": var_seam_iou_low.set(str(v))
            elif k == "seam_band_factor": var_seam_band.set(str(v))
            elif k == "seam_weight": var_seam_weight.set(str(v))
            elif k == "margin_weight": var_margin_weight.set(str(v))
        apply_and_stay(msg="[ADV] Anti-seam preset applied.")

    tk.Button(ar, text="Apply", command=apply_and_stay).pack(side="right")
    tk.Button(ar, text="OK", command=ok_and_close).pack(side="right", padx=6)
    tk.Button(ar, text="Cancel", command=cancel).pack(side="right")
    tk.Button(ar, text="Apply anti-seam dedup", command=apply_anti_seam).pack(side="left")

    # keep dialog above main (initially)
    win.lift()
    try:
        win.attributes("-topmost", True)
        win.after(250, lambda: win.attributes("-topmost", False))
    except Exception:
        pass

    # -- internal: apply built-in preset --
    def _apply_builtin(name: str):
        base = _BUILTIN_PRESETS.get(name)
        if not base:
            set_help("preset"); return
        # fill fields
        var_tile.set(str(base["tile"]))
        var_ovl.set(str(base["overlap"]))
        var_conf.set(str(base["conf"]))
        var_iou.set(str(base["iou_nms"]))
        # pair with fusion & seam defaults
        var_use_wbf.set(bool(_DEFAULT_FUSION["use_wbf"]))
        var_wbf_auto.set(bool(_DEFAULT_FUSION["wbf_auto"]))
        var_wbf_iou.set("")  # auto by default
        var_wbf_alp.set(str(_DEFAULT_FUSION["wbf_alpha"]))
        var_seam_iou_low.set(str(_DEFAULT_SEAM["seam_iou_low"]))
        var_seam_band.set(str(_DEFAULT_SEAM["seam_band_factor"]))
        var_seam_weight.set(str(_DEFAULT_SEAM["seam_weight"]))
        var_margin_weight.set(str(_DEFAULT_SEAM["margin_weight"]))
        apply_and_stay(msg=f"[ADV] Built-in preset applied: {name}")

# ---------- helpers (UI rows & preset I/O) ----------
def _row(parent, label, var: tk.StringVar, focus_cb=None):
    f = tk.Frame(parent); f.pack(fill="x", padx=6, pady=3)
    tk.Label(f, text=label, width=24, anchor="w").pack(side="left")
    e = tk.Entry(f, textvariable=var, width=12)
    e.pack(side="left")
    if focus_cb:
        e.bind("<FocusIn>", lambda _e: focus_cb())

def _binds_to_dict(*vars_) -> dict:
    (var_tile,var_ovl,var_conf,var_iou,var_use_wbf,var_wbf_auto,var_wbf_iou,var_wbf_alp,
     var_seam_iou_low,var_seam_band,var_seam_weight,var_margin_weight) = vars_
    return {
        "tile": var_tile.get(),
        "overlap": var_ovl.get(),
        "conf": var_conf.get(),
        "iou_nms": var_iou.get(),
        "use_wbf": var_use_wbf.get(),
        "wbf_auto": var_wbf_auto.get(),
        "wbf_iou": var_wbf_iou.get(),
        "wbf_alpha": var_wbf_alp.get(),
        "seam_iou_low": var_seam_iou_low.get(),
        "seam_band_factor": var_seam_band.get(),
        "seam_weight": var_seam_weight.get(),
        "margin_weight": var_margin_weight.get(),
    }

def _add_help_bindings(widget, on_focus):
    try:
        widget.bind("<FocusIn>", lambda _e: on_focus())
    except Exception:
        pass

def _import_preset(win, app, cur_vals: dict):
    fp = filedialog.askopenfilename(
        title="Load advanced preset",
        filetypes=[("JSON files","*.json"), ("All files","*.*")],
        parent=win
    )
    if not fp:
        return
    try:
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Preset JSON must be an object with parameter keys.")
        merged = {k: data[k] for k in _PARAM_KEYS if k in data}
        _apply_to_app(app, merged, log_msg=f"[ADV] Preset loaded: {Path(fp).name}")
        messagebox.showinfo("Advanced", f"Preset loaded:\n{fp}", parent=win)
    except Exception as e:
        messagebox.showerror("Advanced", f"Failed to load preset:\n{e}", parent=win)

def _export_preset(win, app, cur_vals: dict):
    # Compose a clean dict with current values (coerce like Apply would)
    to_save = {
        "tile": _int(cur_vals["tile"], 1024),
        "overlap": _float(cur_vals["overlap"], 0.3),
        "conf": _float(cur_vals["conf"], 0.5),
        "iou_nms": _float(cur_vals["iou_nms"], 0.55),
        "use_wbf": bool(cur_vals["use_wbf"].get() if hasattr(cur_vals["use_wbf"], "get") else cur_vals["use_wbf"]),
        "wbf_auto": bool(cur_vals["wbf_auto"].get() if hasattr(cur_vals["wbf_auto"], "get") else cur_vals["wbf_auto"]),
        "wbf_iou": None if (cur_vals["wbf_iou"] in ("", None)) else _float(cur_vals["wbf_iou"], 0.6),
        "wbf_alpha": _float(cur_vals["wbf_alpha"], 0.2),
        "seam_iou_low": _float(cur_vals["seam_iou_low"], 0.30),
        "seam_band_factor": _float(cur_vals["seam_band_factor"], 0.10),
        "seam_weight": _float(cur_vals["seam_weight"], 0.35),
        "margin_weight": _float(cur_vals["margin_weight"], 0.25),
    }
    default_dir = Path.cwd() / "presets"
    default_dir.mkdir(parents=True, exist_ok=True)
    fp = filedialog.asksaveasfilename(
        title="Save advanced preset",
        defaultextension=".json",
        initialdir=str(default_dir),
        initialfile="cv_counter_preset.json",
        filetypes=[("JSON files","*.json")],
        parent=win
    )
    if not fp:
        return
    try:
        Path(fp).write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
        messagebox.showinfo("Advanced", f"Preset saved:\n{fp}", parent=win)
    except Exception as e:
        messagebox.showerror("Advanced", f"Failed to save preset:\n{e}", parent=win)
