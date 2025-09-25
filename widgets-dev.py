# widgets.py — ScrollableFrame (wheel + dynamic width), AOIEditor (multi-AOI), ProgressCanvas
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

class ScrollableFrame(tk.Frame):
    """
    Pionowo przewijany kontener (Frame wewnątrz Canvas + Scrollbar).
    - Obsługa kółka myszy (Windows/macOS/Linux)
    - Dynamiczne dopasowanie szerokości 'inner' do szerokości canvasa także przy <Configure> canvasa
    """
    def __init__(self, parent, height=180, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, height=height, borderwidth=0, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # scrollregion + dopasowanie szerokości na zmiany zawartości
        def _on_inner_config(_e=None):
            try:
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                self.canvas.itemconfigure(self._win, width=self.canvas.winfo_width())
            except Exception:
                pass
        self.inner.bind("<Configure>", _on_inner_config)

        # dopasowanie szerokości przy zmianie szerokości canvasa (np. podczas resize okna)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=self.canvas.winfo_width()))

        # kółko myszy (włączamy gdy kursor nad wnętrzem)
        self.inner.bind("<Enter>", lambda e: self._bind_wheel())
        self.inner.bind("<Leave>", lambda e: self._unbind_wheel())

    def _bind_wheel(self):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)   # Windows/macOS
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)     # Linux
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_wheel(self):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        try:
            if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                self.canvas.yview_scroll(-3, "units")
            else:
                self.canvas.yview_scroll(3, "units")
        except Exception:
            pass


class ProgressCanvas(tk.Frame):
    """Widoczny zawsze pasek postępu rysowany na Canvas (niezależny od motywów ttk)."""
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
        try:
            pct_val = max(0.0, min(100.0, float(self.var.get() or 0.0)))
        except Exception:
            pct_val = 0.0
        pct = pct_val / 100.0
        self.canvas.create_rectangle(1, 1, int(pct*(w-2)), h-1, fill="#3a86ff", outline="#3a86ff", tags="bar")
        self.canvas.create_text(w-40, h//2, text=f"{pct_val:5.1f}%", anchor="c", fill="#222", tags="bar")


class AOIEditor(tk.Frame):
    """
    Prosty edytor wielokątnych AOI (multi-AOI, nazwy).
    API:
      - load_image(path), set_aois(list[{name, polygon}]), get_aois()
      - Enter: zakończ aktualny wielokąt i nazwij
      - Backspace: cofnij ostatni wierzchołek
    """
    def __init__(self, parent, width=900, height=650, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._img = None
        self._img_tk = None
        self._scale = 1.0
        self._offset = (0, 0)
        self._aois = []          # [{name:str, polygon:[(x,y),...]}]
        self._current = []       # [(x,y), ...] — w trakcie rysowania
        self._selected_idx = None
        self._fit_pending = False

        # layout
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

    def start_new_aoi(self):
        self._current = []; self._redraw()

    def delete_selected_aoi(self):
        if self._selected_idx is not None and 0 <= self._selected_idx < len(self._aois):
            del self._aois[self._selected_idx]
            self._selected_idx = None
            self._refresh_list(); self._redraw()

    def clear_all(self):
        self._aois = []; self._current = []; self._selected_idx = None
        self._refresh_list(); self._redraw()

    # --- internals ---
    def _maybe_fit(self):
        if not self._fit_pending or self._img is None: return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        iw, ih = self._img.size
        scale = min(cw / iw, ch / ih, 1.0)
        self._scale = max(1e-6, scale)
        self._offset = ((cw - scale*iw)*0.5, (ch - scale*ih)*0.5)
        from PIL import ImageTk as _ImageTk
        disp = self._img.resize((int(iw*scale), int(ih*scale)))
        self._img_tk = _ImageTk.PhotoImage(disp)
        self._fit_pending = False
        self._redraw()

    def _img_to_disp(self, x, y):
        ox, oy = self._offset
        return (ox + x*self._scale, oy + y*self._scale)

    def _disp_to_img(self, x, y):
        ox, oy = self._offset
        return ((x - ox)/self._scale, (y - oy)/self._scale)

    def _on_select(self):
        sel = self.listbox.curselection()
        self._selected_idx = (sel[0] if sel else None)
        self._redraw()

    def _on_click(self, e):
        if self._img is None: return
        xi, yi = self._disp_to_img(e.x, e.y)
        self._current.append((xi, yi))
        self._redraw()

    def _undo(self, _=None):
        if self._current:
            self._current.pop(); self._redraw()

    def _finish_polygon(self, _=None):
        if not self._current or len(self._current) < 3:
            return
        name = simpledialog.askstring("AOI name", "Name this AOI:", parent=self)
        if not name: name = f"AOI {len(self._aois)+1}"
        self._aois.append({"name": name, "polygon": list(self._current)})
        self._current = []
        self._refresh_list(); self._redraw()

    def _refresh_list(self):
        self.listbox.delete(0, "end")
        for i, a in enumerate(self._aois):
            self.listbox.insert("end", f"{i+1}. {a['name']} ({len(a['polygon'])} pts)")

    def _redraw(self):
        self.canvas.delete("all")
        # tło
        if self._img_tk is not None:
            self.canvas.create_image(self._offset[0], self._offset[1], anchor="nw", image=self._img_tk)
        # istniejące AOI
        for i, a in enumerate(self._aois):
            pts = [self._img_to_disp(x,y) for (x,y) in a["polygon"]]
            color = "#00ff99" if i == self._selected_idx else "#00ffaa"
            if len(pts) >= 2:
                for p, q in zip(pts, pts[1:]+pts[:1]):
                    self.canvas.create_line(p[0], p[1], q[0], q[1], fill=color, width=2, tags="a")
            for (x,y) in pts:
                self.canvas.create_oval(x-3,y-3,x+3,y+3, fill=color, outline="", tags="a")
            # podpis
            if pts:
                x0,y0 = pts[0]
                self.canvas.create_text(x0+8, y0, text=a["name"], anchor="w", fill="#ffffff")
        # rysowany wielokąt
        cur = [self._img_to_disp(x,y) for (x,y) in self._current]
        for p, q in zip(cur, cur[1:]):
            self.canvas.create_line(p[0], p[1], q[0], q[1], fill="#ffe066", width=2)
        for (x,y) in cur:
            self.canvas.create_oval(x-3,y-3,x+3,y+3, fill="#ffe066", outline="")
