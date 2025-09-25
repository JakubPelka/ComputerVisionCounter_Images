# widgets.py — ScrollableFrame (auto-hide vscroll) + AOIEditor (Ctrl+Enter/Backspace)
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2, numpy as np

class ScrollableFrame(tk.Frame):
    """Canvas + inner frame; vertical scrollbar auto-hides when content fits."""
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


class AOIEditor(tk.Frame):
    """Polygon editor:
       • Click to add vertices (max 30)
       • Ctrl+Enter = finish & name
       • Ctrl+Backspace = undo last vertex
       get_aois() returns [{'name': ..., 'polygon': [[x,y],...]}].
    """
    def __init__(self, parent, on_change=None):
        super().__init__(parent)
        self.on_change = on_change
        self.cur_pts = []
        self.polys = []  # [{'name': str, 'pts': [[x,y],...]}]
        self.img_bgr = None
        self.scale = 1.0
        self._tk_img = None
        self._img_node = None

        self.canvas = tk.Canvas(self, bg="#222"); self.canvas.pack(fill="both", expand=True)
        tk.Label(self, text="Finish: Ctrl+Enter   •   Undo vertex: Ctrl+Backspace",
                 fg="#666").pack(fill="x", pady=(4,0))

        self.canvas.bind("<Button-1>", self._on_click)
        self.bind_all("<Control-Return>", self._on_finish, add="+")
        self.bind_all("<Control-BackSpace>", self._on_undo, add="+")

    # public API
    def load_image(self, path: str):
        bgr = cv2.imread(path)
        if bgr is None:
            bgr = np.zeros((720,1280,3), np.uint8)
        self.img_bgr = bgr
        self._fit_base()
        self.cur_pts.clear()
        self._redraw()

    def set_aois(self, aois):
        # accept both 'polygon' and legacy 'pts'
        self.polys = []
        for a in (aois or []):
            name = a.get("name","AOI")
            pts = a.get("polygon", a.get("pts", []))
            self.polys.append({"name": name, "pts": [[float(x),float(y)] for x,y in pts]})
        self.cur_pts.clear()
        self._redraw(); self._notify()

    def get_aois(self):
        # return with 'polygon' key (compatible with counting & masks)
        return [{"name": a["name"], "polygon": [list(p) for p in a["pts"]]} for a in self.polys]

    # internals
    def _fit_base(self):
        H,W = self.img_bgr.shape[:2]
        cw = max(1, self.winfo_width() or 1280)
        ch = max(1, self.winfo_height() or 820)
        s = min(1.0, cw/float(W), ch/float(H))
        self.scale = s
        disp = cv2.resize(self.img_bgr, (int(W*s), int(H*s)))
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
            self._redraw()

    def _on_undo(self, _=None):
        if self.cur_pts:
            self.cur_pts.pop(); self._redraw()

    def _on_finish(self, _=None):
        if len(self.cur_pts) < 3: return
        self._ask_name_and_add(self.cur_pts[:])
        self.cur_pts.clear(); self._redraw()

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

        # center dialog over the editor
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
            self._notify()

    def _notify(self):
        if callable(self.on_change):
            try: self.on_change(self.get_aois())
            except Exception: pass

    def _redraw(self):
        self.canvas.delete("overlay")
        if self._tk_img is not None:
            self.canvas.create_image(0,0,anchor="nw",image=self._tk_img, tags="overlay_base")
        # existing polygons
        for a in self.polys:
            pts = [self._img_to_disp(x,y) for x,y in a["pts"]]
            if len(pts) >= 3:
                for i in range(len(pts)):
                    x1,y1 = pts[i]; x2,y2 = pts[(i+1)%len(pts)]
                    self.canvas.create_line(x1,y1,x2,y2, fill="#FFB000", width=2, tags="overlay")
        # current sketch
        for i, p in enumerate(self.cur_pts):
            x,y = self._img_to_disp(p[0], p[1])
            self.canvas.create_oval(x-3,y-3,x+3,y+3, fill="yellow", outline="", tags="overlay")
            if i>0:
                px,py = self._img_to_disp(self.cur_pts[i-1][0], self.cur_pts[i-1][1])
                self.canvas.create_line(px,py,x,y, fill="yellow", width=2, tags="overlay")
