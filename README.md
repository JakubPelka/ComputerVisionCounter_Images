# ComputerVision Counter — Count anything without coding

A desktop app for **counting objects in images** with your own YOLO models — **no coding required**. Add images, pick classes, (optionally) draw Areas of Interest (AOIs), and let the app count and annotate. Works offline and installs all Python packages into a **local `pkgs/` folder** so it won’t touch your system environment.

> **Status**: Stable for YOLO **`.pt`** models (Ultralytics). **ONNX** is present but **experimental**; support is improving. Segmentation models are planned next.

---

## ✨ Highlights

* **No‑code counting** with your own YOLO models (v5–v8 `.pt`; ONNX experimental).
* **AOI editor** (multi‑polygon per image, named zones). Draw once, auto‑save per image.
* **Flexible counting modes**: by **centroid** (default) or by **box‑overlap** fraction.
* **Per‑class filters**: turn classes on/off; require at least one selected.
* **Advanced presets** (Fast/Balanced/Ultra) + full expert panel (tiling, NMS/WBF, seam de‑dup, etc.).
* **Strict confidence thresholding**: detections must be **strictly greater than** the chosen `conf` value (no rounding).
* **Beautiful overlays**: boxes/labels or centroids, plus **bottom‑right per‑class totals**.
* **GIS‑friendly CSV export** for georeferenced images (points & boxes).
* **Full detections CSV**: run‑level `detections_full.csv` with all kept detections.
* **Local, portable install**: packages go to `./pkgs` or `./_pkgs`.

---

## 🖥️ System requirements

* **OS**: Windows 10/11 (primary target). macOS/Linux likely fine but untested.
* **Python**: 3.10–3.12 (tested with 3.12).
* **GPU**: Optional. If CUDA is available, the app will auto‑use it; otherwise CPU fallback.

> **Models are not included.** Bring your own YOLO weights (`.pt` recommended for now; `.onnx` experimental).

---

## 📦 Installation (local, isolated)

1. **Clone or unzip** this project.
2. (Optional) Create/activate a virtual environment.
3. Run:

   ```bash
   python bootstrap_env.py
   ```

   This installs all required packages into `./pkgs` (or `./_pkgs`) and wires `sys.path` accordingly.
4. Start the app:

   ```bash
   python start_app.py
   ```

> The app will create an `./output/` folder next to the scripts (used as the default output path). Input and weights are **not prefilled** — click **Browse…** to choose them.

---

## 📁 Recommended project layout

```
project/
├─ start_app.py
├─ app_core.py
├─ engine_loader.py
├─ legacy_pt_runner.py
├─ ui_panels.py, ui_advanced.py, widgets.py
├─ bootstrap_env.py
├─ pkgs/                 # local site-packages (auto‑created)
├─ weights/              # put your .pt / .onnx here (not required)
├─ input/                # your images (optional convenience)
└─ output/               # results are written here by default
```

---

## 🚀 Quick start (typical workflow)

1. **Open images**: Click **Browse…** (Input) and select a folder, or select specific files.
2. **Select weights**: Click **Browse…** (Model) and choose your `.pt` (recommended) or `.onnx` (experimental).
3. **Pick classes**: The class list populates from your model. Check at least one.
4. **(Optional) Define AOIs**: Click **AOI Editor**.

   * Draw polygons (multi‑AOI supported)
   * **Finish** polygon: **Ctrl+Enter**
   * **Undo last vertex**: **Ctrl+Backspace**
   * **Name zones** when prompted
   * AOIs are **auto‑saved** per image to `input/aoi/<image>.json` and `input/aoi_masks/<image>.png`.
   * The editor can **import** an AOI JSON to review/extend and **export** a new one.
5. **Choose a preset** (Fast/Balanced/Ultra) or open **Advanced** to fine‑tune.
6. Click **Start**. Progress updates per tile. **Abort** cancels safely.

---

## 🧠 Engines & devices (auto)

* The app selects engine/device automatically:

  * **`.pt`** → legacy PyTorch path (Ultralytics YOLO).
  * **`.onnx`** → ONNX Runtime (CPU/GPU where available; **experimental**).
  * **Device**: GPU if CUDA is available; otherwise **CPU**.
* Manual engine/device selectors were removed from the UI to keep it simple.

> If an ONNX model fails with shape/stride/NMS issues, prefer the original `.pt` for now.

---

## 🧭 AOI behavior (important)

* When you close the AOI editor (or navigate images) your polygons are **auto‑persisted** to `input/aoi` and `input/aoi_masks`.
* If AOIs exist for an image, the main app **auto‑enables** “Use AOI” and reuses them next run.
* **AOI modes**:

  * **Centroid**: a detection counts if the **box center** lies inside any AOI.
  * **Box overlap**: counts if AOI overlap area ≥ **`aoi_box_frac`** (e.g., 0.20 = 20%).

**Hotkeys in AOI editor**

* Finish polygon: **Ctrl+Enter**
* Undo last vertex: **Ctrl+Backspace**

**AOI JSON schema (saved in `input/aoi/*.json`)**

```json
{
  "image": "example.jpg",
  "aois": [
    { "name": "Zone A", "polygon": [[x,y], [x,y], ...] },
    { "name": "Zone B", "polygon": [[x,y], ...] }
  ]
}
```

---

## 🧪 Advanced settings (summary)

Open **Advanced** to adjust:

* **Preset**: Fast / Balanced / Ultra (also available as a slider in main UI).
* **Tiling**

  * `tile` (px): inference patch size.
  * `overlap` (0–1): overlap fraction between tiles.
* **Inference thresholds**

  * `conf`: **strict** confidence threshold. A detection passes **only if `conf_raw > conf_threshold`**.
  * `iou_nms`: NMS IoU threshold.
* **WBF (Weighted Box Fusion)**

  * `Use WBF`: fuse overlapping boxes class‑wise.
  * `Auto WBF IoU`: chooses a sensible IoU for your preset; you can override.
  * `WBF IoU`: fusion overlap threshold.
  * `WBF Alpha`: score weighting exponent.
* **Seam de‑dup** (anti‑tile‑edge duplicates)

  * `Seam band factor`: how close to tile edges the penalty applies.
  * `Seam weight`: how strong the penalty is near seams.
* **Overlay**

  * `boxes` / `boxes_conf` (default) / `centroid` and optional centroid dot.

> Tooltips in the Advanced window explain each parameter in user‑friendly language.

---

## 🧾 Outputs

All outputs go to the **Output** path (default `./output/`). The app won’t overwrite — it adds suffixes as needed.

### 1) Annotated previews

* `annotated/<image>_ann.jpg` — detections + labels (class + conf) or centroids, **plus** bottom‑right per‑class totals.

### 2) Per‑image summary

* `results_per_image.csv` — counts per class for each image.
* `results_totals.json` — total counts across all images.

### 3) Full detections (run level)

* `detections_full.csv` — every **kept** detection (after conf + AOI):

  * Columns: `image, cls, conf, x1, y1, x2, y2, cx, cy, in_aoi`
  * Coordinates are **image pixels**.

### 4) GIS‑friendly CSV (for georeferenced images)

* For images with valid georeference (GeoTIFF or worldfile), we also write:

  * `<image>_3857__detections_p.csv` — **points** (centroids)
  * `<image>_3857__detections_b.csv` — **polygons** (box corners)
* Columns include map coordinates so you can add them directly in GIS (QGIS/ArcGIS).
* If an image isn’t georeferenced, CSVs fall back to pixel coordinates.

> GeoJSON export had visibility issues in some GIS apps; CSV export is the current recommended path.

---

## 🔍 Logging & progress

* **Log panel** in the app + `output/run.log` on disk.
* Useful debug lines:

  * `[DEBUG] conf threshold (raw, strict '>')` — shows the exact numeric threshold used.
  * Per‑image kept counts after NMS and conf filtering.
* **Progress bar** is tile‑based and smoothed; ETA shows as you go.

---

## 🧩 Models

* **Recommended**: Ultralytics `.pt` models (YOLOv5–v8).
* **ONNX (experimental)**: Some models may require specific export flags (dynamic shapes, opset, built‑in NMS). If you encounter "stride"/shape errors or empty outputs, switch back to `.pt` for now.
* **Segmentation**: Planned (not yet in the image workflow).

> Models are not bundled. Make sure your model’s classes align with your expectations; the app reads names from the model.

---

## 🛠️ Tips & gotchas

* Ensure at least **one class** is selected before starting.
* If AOIs exist in `input/aoi/`, the app will reuse them automatically.
* **Centroid mode** is usually best for counting discrete objects; use **box overlap** for large/irregular detections.
* For tiled inference, larger `tile` + higher `overlap` improve quality but slow things down.
* If annotation previews look empty while counts exist, check the **confidence** and **overlay mode**.

---

## 🐞 Troubleshooting

* **I see detections with conf < threshold**

  * The app uses **strict `>`** on the **raw** confidence value (no rounding). If you still see smaller numbers, verify the **raw value** in `detections_full.csv` — UI labels round to 2 decimals for display only.
* **Geo CSVs show up but not where I expect**

  * Confirm the image is truly georeferenced and the worldfile/CRS is correct. CSVs are preferred over GeoJSON for compatibility.
* **ONNX model fails (stride/shape/NMS)**

  * This area is still evolving. Prefer `.pt` models for now.

---

## 🤝 Contributing & licensing

* PRs and issues welcome (bugfixes, ONNX improvements, segmentation support, docs).
* Choose a license that fits your distribution (e.g., MIT). If you publish on Gumroad, include license text with the build.

---

## 📣 Credits

Built for GIS/vision practitioners who want **fast, reliable counts** without writing code — and for power users who enjoy the control when they need it.
