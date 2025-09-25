# widgets.py — ScrollableFrame (z obsługą scrolla kółkiem), AOIEditor (scale fix),
#              ProgressCanvas (z procentem – zawsze widoczny)
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

class ScrollableFrame(tk.Frame):
    def __init__(self, parent, height=180, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, height=height, borderwidth=0, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: (
            self.canvas.configure(scrollregion=self.canvas.bbox("all")),
            self.canvas.itemconfigure(self._win, width=self.canvas.winfo_width())
        ))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")
        # scroll kółkiem (Windows)
        def _mw(e): self.canvas.yview_scroll(-1 if e.delta>0 else 1, "units")
        self.inner.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", _mw))
        self.inner.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

class ProgressCanvas(tk.Frame):
    """Always-visible progress bar drawn on Canvas (works regardless of ttk theme)."""
    def __init__(self, parent, var: tk.DoubleVar, height=22, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.var = var
        self.canvas = tk.Canvas(self, height=height, bg="#f0f0f0",
                                highlightthickness=1, highlightbackground="#888")
        self.canvas.pack(fill="x", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        var.trace_add("write", lambda *_: self._redraw())
        self._redraw()

    def _redraw(self):
        self.canvas.delete("bar")
        w = max(1, int(self.canvas.winfo_width()))
        h = max(4, int(self.canvas.winfo_height()))
        pct_val = max(0.0, min(100.0, float(self.var.get() or 0.0)))
        pct = pct_val / 100.0
        self.canvas.create_rectangle(1, 1, int(pct*(w-2)), h-1, fill="#3a86ff", outline="#3a86ff", tags="bar")
        self.canvas.create_text(w-40, h//2, text=f"{pct_val:5.1f}%", anchor="c", fill="#222", tags="bar")

class AOIEditor(tk.Frame):
    """
    Polygon AOI editor (multi-AOI with names).
    API:
      - load_image(path), set_aois(list[{name, polygon}]), get_aois()
      - Enter: finish polygon & name
      - Backspace: undo last vertex
    """
    def __init__(self, parent, width=900, height=650, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._img = None
        self._img_tk = None
        self._scale = 1.0
        self._offset = (0, 0)
        self._aois = []
        self._current = []
        self._selected_idx = None
        self._fit_pending = False  # scale-fix

        left = tk.Frame(self); left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(self); right.pack(side="right", fill="y")

        self.canvas = tk.Canvas(left, width=width, height=height, bg="#222")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", lambda e: self._maybe_fit())

        tk.Label(right, text="AOIs").pack(anchor="w", padx=4, pady=(4,2))
        self.listbox = tk.Listbox(right, height=18); self.listbox.pack(fill="y", padx=4, pady=(0,6))
        self.listbox.bind("<<ListboxSelect>>", lambda e: self._on_select())

        b = tk.Frame(right); b.pack(fill="x", padx=4)
        tk.Button(b, text="New", command=self.start_new_aoi).pack(fill="x", pady=2)
        tk.Button(b, text="Delete selected", command=self.delete_selected_aoi).pack(fill="x", pady=2)
        tk.Button(b, text="Clear all", command=self.clear_all).pack(fill="x", pady=2)

        tip = "Click to add vertices. Backspace = undo last. Enter = finish polygon + name it."
        tk.Label(right, text=tip, wraplength=220, fg="#666").pack(anchor="w", padx=4, pady=(8,2))

        self.canvas.bind_all("<Return>", self._finish_polygon)
        self.canvas.bind_all("<BackSpace>", self._undo)

    # --- public API ---
    def load_image(self, path: str):
        if Image is None:
            messagebox.showerror("AOI", "Pillow (PIL) is required for AOI editor."); return
        from PIL import Image as _Image, ImageTk as _ImageTk
        try:
            img = _Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("AOI", f"Cannot open image:\n{e}"); return
        self._img = img
        self._fit_pending = True
        self.after(30, self._maybe_fit)

    def set_aois(self, aois):
        self._aois = []
        for a in (aois or []):
            poly = [(float(x), float(y)) for x,y in a.get("polygon", [])]
            self._aois.append({"name": str(a.get("name","AOI")), "polygon": poly})
        self._refresh_list(); self._redraw()

    def get_aois(self):
        return [{"name": a["name"], "polygon": list(a["polygon"])} for a in self._aois]

    def start_new_aoi(self): self._current = []; self._redraw()
    def delete_selected_aoi(self):
        if self._selected_idx is not None and 0 <= self._selected_idx < len(self._aois):
            del self._aois[self._selected_idx]; self._selected_idx = None; self._refresh_list(); self._redraw()
    def clear_all(self): self._aois = []; self._current = []; self._selected_idx = None; self._refresh_list(); self._redraw()

    # --- internals ---
    def _maybe_fit(self):
        if self._img is None or not self._fit_pending: return
        cw = int(self.canvas.winfo_width() or 0); ch = int(self.canvas.winfo_height() or 0)
        if cw <= 2 or ch <= 2: return
        self._fit(); self._fit_pending = False; self._current=[]; self._selected_idx=None; self._redraw()

    def _fit(self):
        if not self._img: return
        cw = max(1, int(self.canvas.winfo_width() or self.canvas.cget("width")))
        ch = max(1, int(self.canvas.winfo_height() or self.canvas.cget("height")))
        iw, ih = self._img.size
        s = max(0.01, min(4.0, min(cw/iw, ch/ih)))
        self._scale = s
        nw, nh = int(iw*s), int(ih*s)
        self._img_tk = ImageTk.PhotoImage(self._img.resize((nw, nh)))
        self._offset = ((cw - nw)//2, (ch - nh)//2)

    def _img_to_canvas(self, x, y):
        ox, oy = self._offset
        return ox + x*self._scale, oy + y*self._scale

    def _canvas_to_img(self, x, y):
        ox, oy = self._offset
        return (x-ox)/self._scale, (y-oy)/self._scale

    def _refresh_list(self):
        self.listbox.delete(0, "end")
        for i, a in enumerate(self._aois):
            self.listbox.insert("end", f"{i+1}. {a['name']} ({len(a['polygon'])} pts)")

    def _on_select(self):
        sel = self.listbox.curselection()
        self._selected_idx = (int(sel[0]) if sel else None)
        self._redraw()

    def _on_click(self, e):
        if not self._img: return
        x, y = self._canvas_to_img(e.x, e.y)
        self._current.append((x,y))
        self._redraw()

    def _undo(self, _=None):
        if self._current: self._current.pop(); self._redraw()

    def _finish_polygon(self, _=None):
        if len(self._current) < 3: return
        name = simpledialog.askstring("AOI name", "Enter AOI name:", parent=self) or f"AOI {len(self._aois)+1}"
        self._aois.append({"name": name, "polygon": list(self._current)})
        self._current = []; self._selected_idx = len(self._aois)-1
        self._refresh_list(); self._redraw()

    def _draw_polygon(self, poly, color="#00ff88", width=2, dash=None):
        if len(poly) < 2: return
        pts = []
        for x,y in poly:
            cx, cy = self._img_to_canvas(x,y); pts.extend([cx,cy])
        self.canvas.create_polygon(pts, outline=color, width=width, dash=dash, fill="", tags="overlay")
        for x,y in poly:
            cx, cy = self._img_to_canvas(x,y)
            self.canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill=color, outline="", tags="overlay")

    def _redraw(self):
        self.canvas.delete("all")
        if self._img_tk:
            ox, oy = self._offset
            self.canvas.create_image(ox, oy, image=self._img_tk, anchor="nw")
        for i,a in enumerate(self._aois):
            sel = (i == self._selected_idx)
            self._draw_polygon(a["polygon"], "#00e0ff" if sel else "#00ff88", 3 if sel else 2)
        if self._current:
            self._draw_polygon(self._current, "#ffcc00", 2, dash=(3,2))
