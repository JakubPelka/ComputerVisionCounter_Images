# ui_advanced.py — Advanced options dialog (dedup preset included)
from __future__ import annotations
import tkinter as tk
from tkinter import messagebox

DEFAULTS = dict(
    DEFAULT_WBF_ALPHA=0.20,
    DEFAULT_SEAM_IOU_LOW=0.30,
    DEFAULT_SEAM_BAND_FACTOR=0.10,
    DEFAULT_SEAM_WEIGHT=0.35,
    DEFAULT_MARGIN_WEIGHT=0.25,
)

def auto_wbf_iou(q, nms): return 0.60 if int(q) == 5 else max(0.55, float(nms))

def open_advanced(app):
    QUALITY_PRESETS = app.__dict__.get("QUALITY_PRESETS", None) or {
        1: {"tile": 640,  "overlap": 0.15, "conf": 0.40, "iou_nms": 0.65, "use_wbf": True},
        2: {"tile": 896,  "overlap": 0.25, "conf": 0.45, "iou_nms": 0.60, "use_wbf": True},
        3: {"tile": 1024, "overlap": 0.30, "conf": 0.50, "iou_nms": 0.55, "use_wbf": True},
        4: {"tile": 1280, "overlap": 0.45, "conf": 0.60, "iou_nms": 0.50, "use_wbf": True},
        5: {"tile": 2560, "overlap": 0.60, "conf": 0.75, "iou_nms": 0.40, "use_wbf": True},
    }
    DEF = DEFAULTS

    win = tk.Toplevel(app); win.title("Advanced options"); win.geometry("560x380")
    frame = tk.Frame(win); frame.pack(fill="both", expand=True, padx=8, pady=8)

    # Left: slider + label. Right: fields + actions.
    left = tk.Frame(frame); left.pack(side="left", fill="both", expand=True)
    right = tk.Frame(frame); right.pack(side="right", fill="y")

    p = QUALITY_PRESETS.get(int(app.quality.get()), QUALITY_PRESETS[5])
    tk.Label(left, text="Quality (1=faster, 5=ULTRA)").pack(anchor="w")
    sc = tk.Scale(left, from_=1, to=5, orient="horizontal", variable=app.quality, showvalue=True,
                  command=lambda _=None: app._update_preset_label(), length=260)
    sc.pack(anchor="w")
    preset_txt = tk.StringVar(value=f"tile={p['tile']} overlap={p['overlap']} conf={p['conf']} nms={p['iou_nms']} WBF={p['use_wbf']}")
    lbl = tk.Label(left, textvariable=preset_txt, fg="#555"); lbl.pack(anchor="w", pady=(6,0))
    def _refresh_label(*_):
        pp = QUALITY_PRESETS.get(int(app.quality.get()), QUALITY_PRESETS[5])
        preset_txt.set(f"tile={pp['tile']} overlap={pp['overlap']} conf={pp['conf']} nms={pp['iou_nms']} WBF={pp['use_wbf']}")
    app.quality.trace_add("write", _refresh_label)

    def row(lbl, var):
        f = tk.Frame(right); f.pack(fill="x", pady=2, anchor="e")
        tk.Label(f, text=lbl, width=22, anchor="e").pack(side="left")
        tk.Entry(f, textvariable=var, width=10).pack(side="left"); return f

    def current_auto_wbf():
        pp = QUALITY_PRESETS.get(int(app.quality.get()), QUALITY_PRESETS[5])
        return auto_wbf_iou(int(app.quality.get()), float(pp["iou_nms"]))

    base = app.advanced_params if app.advanced_override else {
        **p, "wbf_alpha": DEF["DEFAULT_WBF_ALPHA"], "wbf_iou": None, "wbf_auto": True,
        "seam_iou_low": DEF["DEFAULT_SEAM_IOU_LOW"], "seam_band_factor": DEF["DEFAULT_SEAM_BAND_FACTOR"],
        "seam_weight": DEF["DEFAULT_SEAM_WEIGHT"], "margin_weight": DEF["DEFAULT_MARGIN_WEIGHT"]
    }
    S = lambda k, d="": tk.StringVar(value=str(base.get(k, d)))
    var_tile, var_ov, var_conf, var_nms = S("tile"), S("overlap"), S("conf"), S("iou_nms")
    var_wbf = tk.BooleanVar(value=bool(base.get("use_wbf", True)))
    var_alpha = S("wbf_alpha", DEF["DEFAULT_WBF_ALPHA"])
    var_wbf_auto = tk.BooleanVar(value=bool(base.get("wbf_auto", True)))
    var_wbf_iou = tk.StringVar(value=str(current_auto_wbf()) if var_wbf_auto.get() else ("" if base.get("wbf_iou", None) is None else str(base["wbf_iou"])))
    var_seam_iou_low, var_seam_band, var_seam_w, var_margin_w = S("seam_iou_low", DEF["DEFAULT_SEAM_IOU_LOW"]), S("seam_band_factor", DEF["DEFAULT_SEAM_BAND_FACTOR"]), S("seam_weight", DEF["DEFAULT_SEAM_WEIGHT"]), S("margin_weight", DEF["DEFAULT_MARGIN_WEIGHT"])

    for lbltxt, var in [("Tile size (px)", var_tile), ("Overlap (0..1)", var_ov), ("Confidence", var_conf), ("IoU NMS", var_nms)]:
        row(lbltxt, var)
    tk.Checkbutton(right, text="Use WBF dedup", variable=var_wbf).pack(anchor="e")
    rf = tk.Frame(right); rf.pack(anchor="e")
    tk.Label(rf, text="WBF alpha", width=22, anchor="e").pack(side="left"); tk.Entry(rf, textvariable=var_alpha, width=10).pack(side="left")
    r2 = tk.Frame(right); r2.pack(anchor="e")
    tk.Checkbutton(r2, text="Auto WBF IoU", variable=var_wbf_auto,
                   command=lambda: ent_wbf.config(state=("disabled" if var_wbf_auto.get() else "normal"))).pack(side="left")
    ent_wbf = tk.Entry(r2, textvariable=var_wbf_iou, width=10); ent_wbf.pack(side="left")
    ent_wbf.config(state=("disabled" if var_wbf_auto.get() else "normal"))
    for lbltxt, var in [("Seam low IoU", var_seam_iou_low), ("Seam band factor", var_seam_band), ("Weight margin", var_margin_w), ("Weight seam dist", var_seam_w)]:
        row(lbltxt, var)

    btns = tk.Frame(win); btns.pack(fill="x", pady=8)
    def apply_override():
        try:
            cur = QUALITY_PRESETS.get(int(app.quality.get()), QUALITY_PRESETS[5])
            def get_or(v, cast, key): s=v.get().strip(); return cast(s) if s!="" else cast(cur[key])
            app.advanced_params = {
                "tile": get_or(var_tile, int, "tile"),
                "overlap": get_or(var_ov, float, "overlap"),
                "conf": get_or(var_conf, float, "conf"),
                "iou_nms": get_or(var_nms, float, "iou_nms"),
                "use_wbf": bool(var_wbf.get()),
                "wbf_alpha": float(var_alpha.get()) if var_alpha.get().strip()!="" else DEF["DEFAULT_WBF_ALPHA"],
                "wbf_iou": (None if var_wbf_auto.get() else (float(var_wbf_iou.get()) if var_wbf_iou.get().strip()!="" else None)),
                "wbf_auto": bool(var_wbf_auto.get()),
                "seam_iou_low": float(var_seam_iou_low.get()) if var_seam_iou_low.get().strip()!="" else DEF["DEFAULT_SEAM_IOU_LOW"],
                "seam_band_factor": float(var_seam_band.get()) if var_seam_band.get().strip()!="" else DEF["DEFAULT_SEAM_BAND_FACTOR"],
                "seam_weight": float(var_seam_w.get()) if var_seam_w.get().strip()!="" else DEF["DEFAULT_SEAM_WEIGHT"],
                "margin_weight": float(var_margin_w.get()) if var_margin_w.get().strip()!="" else DEF["DEFAULT_MARGIN_WEIGHT"],
            }
            app.advanced_override = True
            app._log("[ADV] Override enabled."); win.destroy()
        except Exception as e:
            messagebox.showerror("Advanced", str(e))
    def reset_to_preset():
        app.advanced_override = False; app._log("[ADV] Preset from slider restored."); win.destroy()
    def apply_anti_seam():
        app.advanced_params = {
            "tile": int(var_tile.get()) if var_tile.get().strip() else QUALITY_PRESETS[int(app.quality.get())]["tile"],
            "overlap": 0.55, "conf": float(var_conf.get() or 0.5),
            "iou_nms": float(var_nms.get() or 0.55),
            "use_wbf": True, "wbf_alpha": 0.35, "wbf_iou": 0.60, "wbf_auto": False,
            "seam_iou_low": 0.40, "seam_band_factor": 0.12, "seam_weight": 0.45, "margin_weight": 0.30,
        }
        app.advanced_override = True; app._log("[ADV] Anti-seam preset applied."); win.destroy()

    tk.Button(btns, text="Apply override", command=apply_override).pack(side="left", padx=6)
    tk.Button(btns, text="Use preset", command=reset_to_preset).pack(side="left", padx=6)
    tk.Button(btns, text="Anti-seam (dedup)", command=apply_anti_seam).pack(side="right", padx=6)
