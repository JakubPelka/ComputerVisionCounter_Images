
# widgets.py — ScrollableFrame and AOIEditor (compatible 1:1 with start_app/ui_panels)
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from pathlib import Path

try:
    from PIL import Image, ImageTk, ImageDraw
except Exception as e:
    Image = None
    ImageTk = None

class ScrollableFrame(tk.Frame):
    """A vertically scrollable frame (common utility)."""
    def __init__(self, parent, height=180, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas)

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")


class AOIEditor(tk.Frame):
    """
    Simple polygon AOI editor:
      - Click to add points
      - Backspace to undo last point
      - Enter to finish polygon (you'll be asked to name it)
      - Supports multiple AOIs per image; list shows on the right
      - Buttons: New AOI, Delete selected, Clear all
    Methods used by app:
      - load_image(path:str)
      - set_aois(list_of_dicts[{name, polygon: [(x,y),...]}])
      - get_aois() -> list_of_dicts
      - start_new_aoi(), delete_selected_aoi(), clear_all()
    """
    def __init__(self, parent, width=900, height=650, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.configure(bg=self.cget("bg"))
        self._img_path = None
        self._img = None           # PIL.Image
        self._img_tk = None        # ImageTk.PhotoImage
        self._scale = 1.0
        self._offset = (0, 0)      # left, top inside canvas for centering
        self._aois = []            # [{'name': str, 'polygon': [(x,y),...]}]
        self._current = []         # building polygon (display coords)
        self._selected_idx = None

        # Layout: canvas + right panel
        left = tk.Frame(self); left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(self); right.pack(side="right", fill="y")

        # Canvas with fixed size; scales image to fit
        self.canvas = tk.Canvas(left, width=width, height=height, bg="#222")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        # Right panel: list + buttons
        tk.Label(right, text="AOIs").pack(anchor="w", padx=4, pady=(4,2))
        self.listbox = tk.Listbox(right, height=18)
        self.listbox.pack(fill="y", padx=4, pady=(0,6))
        self.listbox.bind("<<ListboxSelect>>", lambda e: self._on_select_list())

        bt = tk.Frame(right); bt.pack(fill="x", padx=4)
        tk.Button(bt, text="New", command=self.start_new_aoi).pack(fill="x", pady=2)
        tk.Button(bt, text="Delete selected", command=self.delete_selected_aoi).pack(fill="x", pady=2)
        tk.Button(bt, text="Clear all", command=self.clear_all).pack(fill="x", pady=2)

        tip = ("Click to add vertices. Backspace = undo last point. Enter = finish polygon and name it.\n"
               "You can create multiple AOIs; select any in the list to highlight.")
        tk.Label(right, text=tip, wraplength=220, fg="#666").pack(anchor="w", padx=4, pady=(8,2))

        # Key bindings
        self.canvas.bind_all("<Return>", self._finish_polygon)
        self.canvas.bind_all("<BackSpace>", self._undo_point)

    # ---- Public API -----------------------------------------------------------
    def load_image(self, path: str):
        self._img_path = str(path)
        if Image is None:
            messagebox.showerror("AOI", "Pillow (PIL) is required for AOI editor.")
            return
        try:
            img = Image.open(self._img_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("AOI", f"Cannot open image:\n{e}")
            return
        self._img = img
        self._current = []
        self._selected_idx = None
        self._fit_image_to_canvas()
        self._redraw()

    def set_aois(self, aois):
        """aois: list of {'name': str, 'polygon': [(x,y), ...]} in image pixel coordinates"""
        self._aois = []
        for a in (aois or []):
            poly = [(float(x), float(y)) for x,y in a.get("polygon", [])]
            self._aois.append({"name": str(a.get("name","AOI")), "polygon": poly})
        self._refresh_list()
        self._redraw()

    def get_aois(self):
        """Return AOIs in image pixel coordinates."""
        return [{"name": a["name"], "polygon": list(a["polygon"])} for a in self._aois]

    def start_new_aoi(self):
        self._current = []
        self._selected_idx = None
        self._redraw()

    def delete_selected_aoi(self):
        if self._selected_idx is None or self._selected_idx >= len(self._aois):
            return
        del self._aois[self._selected_idx]
        self._selected_idx = None
        self._refresh_list()
        self._redraw()

    def clear_all(self):
        self._aois = []
        self._current = []
        self._selected_idx = None
        self._refresh_list()
        self._redraw()

    # ---- Internal helpers -----------------------------------------------------
    def _fit_image_to_canvas(self):
        if not self._img: return
        cw = max(1, int(self.canvas.winfo_width() or self.canvas.cget("width")))
        ch = max(1, int(self.canvas.winfo_height() or self.canvas.cget("height")))
        iw, ih = self._img.size
        s = min(cw / iw, ch / ih)
        s = max(0.01, min(4.0, s))
        self._scale = s
        nw, nh = int(iw * s), int(ih * s)
        if Image:
            disp = self._img.resize((nw, nh), Image.LANCZOS)
            self._img_tk = ImageTk.PhotoImage(disp)
        self._offset = ((cw - nw) // 2, (ch - nh) // 2)

    def _img_to_canvas(self, x, y):
        ox, oy = self._offset
        return ox + x*self._scale, oy + y*self._scale

    def _canvas_to_img(self, x, y):
        ox, oy = self._offset
        return (x - ox)/self._scale, (y - oy)/self._scale

    def _refresh_list(self):
        self.listbox.delete(0, "end")
        for i, a in enumerate(self._aois):
            self.listbox.insert("end", f"{i+1}. {a['name']}  ({len(a['polygon'])} pts)")

    def _on_select_list(self):
        sel = self.listbox.curselection()
        if not sel:
            self._selected_idx = None
        else:
            self._selected_idx = int(sel[0])
        self._redraw()

    def _on_click(self, event):
        if not self._img: return
        # Add vertex to current polygon
        x, y = self._canvas_to_img(event.x, event.y)
        self._current.append((x, y))
        self._redraw()

    def _undo_point(self, event=None):
        if self._current:
            self._current.pop()
            self._redraw()

    def _finish_polygon(self, event=None):
        if len(self._current) < 3:
            return
        name = simpledialog.askstring("AOI name", "Enter AOI name:", parent=self)
        if not name:
            name = f"AOI {len(self._aois)+1}"
        # Close polygon
        poly = list(self._current)
        self._aois.append({"name": name, "polygon": poly})
        self._current = []
        self._selected_idx = len(self._aois)-1
        self._refresh_list()
        self._redraw()

    def _draw_polygon(self, poly, outline="#00ff88", width=2, dash=None, fill_transparent=True):
        if len(poly) < 2: return
        pts = []
        for x, y in poly:
            cx, cy = self._img_to_canvas(x, y)
            pts.extend([cx, cy])
        if fill_transparent:
            # semi-transparent fill via stipple-style polygon approximation (tk doesn't do alpha)
            self.canvas.create_polygon(pts, outline=outline, width=width, dash=dash,
                                       fill="", smooth=False)
        else:
            self.canvas.create_polygon(pts, outline=outline, width=width, dash=dash,
                                       fill="", smooth=False)
        # Draw vertices
        for x, y in poly:
            cx, cy = self._img_to_canvas(x, y)
            r = 3
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=outline, outline="")

    def _redraw(self):
        self.canvas.delete("all")
        # draw image
        if self._img and self._img_tk:
            self._fit_image_to_canvas()
            ox, oy = self._offset
            self.canvas.create_image(ox, oy, image=self._img_tk, anchor="nw")
        # draw existing AOIs
        for i, a in enumerate(self._aois):
            sel = (i == self._selected_idx)
            color = "#00e0ff" if sel else "#00ff88"
            self._draw_polygon(a["polygon"], outline=color, width=3 if sel else 2)
            # label
            if a["polygon"]:
                xs = [p[0] for p in a["polygon"]]; ys = [p[1] for p in a["polygon"]]
                cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
                tx, ty = self._img_to_canvas(cx, cy)
                self.canvas.create_text(tx, ty, text=a["name"], fill=color, anchor="s")
        # draw current polygon (dashed)
        if self._current:
            self._draw_polygon(self._current, outline="#ffcc00", width=2, dash=(3,2))
