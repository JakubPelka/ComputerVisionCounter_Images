# widgets.py — ScrollableFrame (auto-hide vscroll) + AOIEditor with AOI Manager
# Changes:
# - Keep AOI editor on top when opening/closing file dialogs (parented + lift/topmost pulse)
# - Add "Clear (current image)" button to toolbar
# - Minor hardening of JSON load/save (same parsing as before)

from __future__ import annotations
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2, numpy as np

# ---------- ScrollableFrame ----------
class ScrollableFrame(tk.Frame):
    def __init__(self, parent, height=260):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, height=height)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")
        self._vbar_visible = True

        self.inner = tk.Frame(self.canvas)
        self.win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        def _on_inner_config(_):
            self.canvas.configure(scrollregion=self.canvas.bbox("all")); self._update_scrollbar()
        def _on_canvas_config(e):
            self.canvas.itemconfig(self.win, width=e.width); self._update_scrollbar()
        self.inner.bind("<Configure>", _on_inner_config)
        self.canvas.bind("<Configure>", _on_canvas_config)

        def _mw(e): self.canvas.yview_scroll(int(-e.delta/120), "units")
        self.inner.bind("<Enter>", lambda _e: self.canvas.bind_all("<MouseWheel>", _mw))
        self.inner.bind("<Leave>", lambda _e: self.canvas.unbind_all("<MouseWheel>"))

    def _update_scrollbar(self):
        try:
            bbox = self.canvas.bbox("all")
            content_h = (bbox[3]-bbox[1]) if bbox else 0
            canvas_h = int(self.canvas.winfo_height())
            need = content_h > canvas_h + 2
        except Exception:
            need = False
        if need and not self._vbar_visible:
            self.vbar.pack(side="right", fill="y"); self._vbar_visible = True
        elif not need and self._vbar_visible:
            self.vbar.pack_forget(); self._vbar_visible = False


# ---------- Helpers ----------
def _parse_aois_json(data) -> list[dict]:
    """Returns [{'name': str, 'polygon': [[x,y],...]}] for multiple legacy formats."""
    out = []
    def _norm(pts): return [[float(x), float(y)] for x, y in pts]

    if isinstance(data, dict):
        if isinstance(data.get("aois"), list):
            for a in data["aois"]:
                pts = a.get("polygon") or a.get("points") or a.get("pts")
                if pts and len(pts) >= 3:
                    out.append({"name": a.get("name","AOI"), "polygon": _norm(pts)})
        else:
            pts = data.get("polygon") or data.get("points") or data.get("pts")
            if pts and len(pts) >= 3:
                out.append({"name": data.get("name","AOI 1"), "polygon": _norm(pts)})
    elif isinstance(data, list) and data:
        if isinstance(data[0], dict):
            for a in data:
                pts = a.get("polygon") or a.get("points") or a.get("pts")
                if pts and len(pts) >= 3:
                    out.append({"name": a.get("name","AOI"), "polygon": _norm(pts)})
        else:
            if len(data) >= 3:
                out.append({"name":"AOI 1", "polygon": _norm(data)})
    return out


# ---------- AOI Editor ----------
class AOIEditor(tk.Frame):
    """Polygon editor with a small AOI Manager toolbar."""
    def __init__(self, parent, on_change=None):
        super().__init__(parent)
        self.on_change = on_change
        self.cur_pts = []
        self.polys = []                 # [{'name': str, 'pts': [[x,y],...]}]
        self.img_bgr = None
        self.img_path: str | None = None
        self.scale = 1.0
        self._tk_img = None
        self._img_node = None
        self._first_fit_done = False
        self._needs_refit = False

        # Toolbar (AOI Manager)
        tb = tk.Frame(self); tb.pack(fill="x", pady=(0,4))
        tk.Button(tb, text="Load JSON…", command=self._load_json_dialog).pack(side="left")
        tk.Button(tb, text="Save JSON as…", command=self._save_json_dialog).pack(side="left", padx=6)
        tk.Button(tb, text="Load from INPUT/aoi", command=self._load_from_input_aoi).pack(side="left")
        tk.Button(tb, text="Clear (current image)", command=self._clear_current).pack(side="left", padx=6)

        # Canvas + hint
        self.canvas = tk.Canvas(self, bg="#222"); self.canvas.pack(fill="both", expand=True)
        tk.Label(self, text="Click to add vertices • Finish: Ctrl+Enter • Undo vertex: Ctrl+Backspace",
                 fg="#666").pack(fill="x", pady=(4,0))

        self.canvas.bind("<Button-1>", self._on_click)
        self.bind_all("<Control-Return>", self._on_finish, add="+")
        self.bind_all("<Control-BackSpace>", self._on_undo, add="+")
        self.bind("<Configure>", self._on_configure, add="+")  # first-open render fix

    # public API
    def load_image(self, path: str):
        self.img_path = path
        bgr = cv2.imread(path)
        if bgr is None:
            bgr = np.zeros((720,1280,3), np.uint8)
        self.img_bgr = bgr
        self._first_fit_done = False
        self._fit_base()           # try immediately
        self.cur_pts.clear()
        self._redraw()

    def set_aois(self, aois):
        self.polys = []
        for a in (aois or []):
            name = a.get("name","AOI")
            pts = a.get("polygon", a.get("pts", []))
            self.polys.append({"name": name, "pts": [[float(x),float(y)] for x,y in pts]})
        self.cur_pts.clear()
        self._redraw(); self._notify()

    def get_aois(self):
        return [{"name": a["name"], "polygon": [list(p) for p in a["pts"]]} for a in self.polys]

    # ----- Toolbar actions -----
    def _dlg_parent(self):
        # Parent all dialogs to the editor's toplevel (keeps modality and z-order)
        return self.winfo_toplevel()

    def _pulse_topmost(self):
        # Briefly force the editor on top so it doesn't fall behind main window after dialogs
        top = self.winfo_toplevel()
        try:
            top.lift()
            top.attributes("-topmost", True)
            top.after(250, lambda: top.attributes("-topmost", False))
        except Exception:
            pass

    def _load_json_dialog(self):
        fp = filedialog.askopenfilename(
            title="Load AOIs from JSON",
            filetypes=[("JSON files","*.json"), ("All files","*.*")],
            parent=self._dlg_parent()
        )
        if not fp: 
            self._pulse_topmost(); 
            return
        try:
            data = json.loads(Path(fp).read_text(encoding="utf-8"))
            aois = _parse_aois_json(data)
            if not aois:
                raise ValueError("No polygons found in file.")
            self.set_aois(aois)
            messagebox.showinfo("AOI", f"Loaded {len(aois)} polygon(s) from:\n{fp}", parent=self._dlg_parent())
        except Exception as e:
            messagebox.showerror("AOI", f"Failed to load JSON:\n{e}", parent=self._dlg_parent())
        finally:
            self._pulse_topmost()

    def _save_json_dialog(self):
        if self.img_path:
            d = Path(self.img_path).parent / "aoi"
            d.mkdir(parents=True, exist_ok=True)
            default = (d / f"{Path(self.img_path).stem}.json")
        else:
            d = Path.cwd(); default = d / "aoi.json"
        fp = filedialog.asksaveasfilename(
            title="Save AOIs to JSON",
            defaultextension=".json",
            initialdir=str(d),
            initialfile=default.name,
            filetypes=[("JSON files","*.json")],
            parent=self._dlg_parent()
        )
        if not fp:
            self._pulse_topmost()
            return
        try:
            out = {
                "image": Path(self.img_path).name if self.img_path else None,
                "aois": [{"name": a["name"],
                          "polygon": [list(p) for p in a["pts"]],
                          "points":  [list(p) for p in a["pts"]]}  # legacy mirror
                         for a in self.polys]
            }
            Path(fp).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            messagebox.showinfo("AOI", f"Saved {len(self.polys)} polygon(s) to:\n{fp}", parent=self._dlg_parent())
        except Exception as e:
            messagebox.showerror("AOI", f"Failed to save JSON:\n{e}", parent=self._dlg_parent())
        finally:
            self._pulse_topmost()

    def _load_from_input_aoi(self):
        if not self.img_path:
            messagebox.showinfo("AOI", "Open an image first.", parent=self._dlg_parent()); 
            self._pulse_topmost(); 
            return
        jf = Path(self.img_path).parent / "aoi" / f"{Path(self.img_path).stem}.json"
        if not jf.exists():
            messagebox.showwarning("AOI", f"No JSON at:\n{jf}", parent=self._dlg_parent()); 
            self._pulse_topmost(); 
            return
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            aois = _parse_aois_json(data)
            if not aois:
                raise ValueError("No polygons found.")
            self.set_aois(aois)
            messagebox.showinfo("AOI", f"Loaded {len(aois)} polygon(s) from:\n{jf}", parent=self._dlg_parent())
        except Exception as e:
            messagebox.showerror("AOI", f"Failed to load JSON:\n{e}", parent=self._dlg_parent())
        finally:
            self._pulse_topmost()

    def _clear_current(self):
        self.polys.clear()
        self.cur_pts.clear()
        self._redraw(); self._notify()

    # ----- internals -----
    def _on_configure(self, _evt=None):
        if self.img_bgr is not None and not self._first_fit_done:
            self._fit_base(); self._redraw(); self._first_fit_done = True
        elif self._needs_refit:
            self._fit_base(); self._redraw(); self._needs_refit = False

    def _fit_base(self):
        if self.img_bgr is None:
            return
        H,W = self.img_bgr.shape[:2]
        self.update_idletasks()
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw < 100 or ch < 100:
            cw, ch = 1280, 820
            self._needs_refit = True
        s = min(1.0, cw/float(W), ch/float(H))
        self.scale = s
        disp = cv2.resize(self.img_bgr, (max(1,int(W*s)), max(1,int(H*s))))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        if self._img_node is None:
            self._img_node = self.canvas.create_image(0,0,anchor="nw",image=self._tk_img)
        else:
            self.canvas.itemconfigure(self._img_node, image=self._tk_img)
        self.canvas.config(width=disp.shape[1], height=disp.shape[0])

    def _img_to_disp(self, x, y): return [x*self.scale, y*self.scale]
    def _disp_to_img(self, x, y): return [x/self.scale, y/self.scale]

    def _on_click(self, e):
        xi, yi = self._disp_to_img(e.x, e.y)
        if len(self.cur_pts) < 30:
            self.cur_pts.append([xi, yi])
            self._redraw(); self._notify()

    def _on_undo(self, _=None):
        if self.cur_pts:
            self.cur_pts.pop(); self._redraw(); self._notify()

    def _on_finish(self, _=None):
        if len(self.cur_pts) < 3: return
        self._ask_name_and_add(self.cur_pts[:])
        self.cur_pts.clear(); self._redraw(); self._notify()

    def _ask_name_and_add(self, pts):
        win = tk.Toplevel(self); win.title("AOI name")
        win.transient(self); win.grab_set(); win.lift()

        var = tk.StringVar(value=f"AOI {len(self.polys)+1}")
        tk.Label(win, text="Zone name:").pack(padx=10, pady=(10,4), anchor="w")
        ent = tk.Entry(win, textvariable=var, width=28); ent.pack(padx=10, pady=(0,10))
        ent.focus_set()
        ent.bind("<Return>", lambda e: ok())

        btns = tk.Frame(win); btns.pack(padx=10, pady=(0,10), fill="x")
        tk.Button(btns, text="OK", command=lambda: ok()).pack(side="right")

        # center dialog
        win.update_idletasks()
        ww, wh = win.winfo_width(), win.winfo_height()
        px, py = self.winfo_rootx(), self.winfo_rooty()
        pw, ph = self.winfo_width(), self.winfo_height()
        x = px + max(0, (pw - ww)//2); y = py + max(0, (ph - wh)//2)
        win.geometry(f"+{x}+{y}")

        out = {"name": None}
        def ok():
            out["name"] = var.get().strip() or f"AOI {len(self.polys)+1}"
            win.destroy()

        self.wait_window(win)
        if out["name"]:
            self.polys.append({"name": out["name"], "pts": [list(p) for p in pts]})

    def _notify(self):
        if callable(self.on_change):
            try: self.on_change(self.get_aois())
            except Exception: pass

    def _redraw(self):
        self.canvas.delete("overlay")
        if self._tk_img is not None:
            self.canvas.create_image(0,0,anchor="nw",image=self._tk_img, tags="overlay_base")
        for a in self.polys:
            pts = [self._img_to_disp(x,y) for x,y in a["pts"]]
            if len(pts) >= 3:
                for i in range(len(pts)):
                    x1,y1 = pts[i]; x2,y2 = pts[(i+1)%len(pts)]
                    self.canvas.create_line(x1,y1,x2,y2, fill="#FFB000", width=2, tags="overlay")
        for i, p in enumerate(self.cur_pts):
            x,y = self._img_to_disp(p[0], p[1])
            self.canvas.create_oval(x-3,y-3,x+3,y+3, fill="yellow", outline="", tags="overlay")
            if i>0:
                px,py = self._img_to_disp(self.cur_pts[i-1][0], self.cur_pts[i-1][1])
                self.canvas.create_line(px,py,x,y, fill="yellow", width=2, tags="overlay")
