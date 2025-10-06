# ComputerVision Counter — Count anything without coding

A desktop app for **counting objects in images** with your own YOLO models — **no coding required**. Add images, pick classes, (optionally) draw Areas of Interest (AOIs), and let the app count and annotate. Works offline and installs all Python packages into a **local **``** folder** so it won’t touch your system environment.

> **Status (this release):** Supports Ultralytics **YOLO **`` models. **ONNX is disabled** for now and returns in the next release.

---

## ✨ Highlights

* **No‑code counting** with your own YOLO `` models.
* **AOI editor** (multi‑polygon per image, named zones). Draw once, auto‑save per image.
* **Flexible AOI filtering**: by **centroid** (default) or **box‑overlap** fraction.
* **AOI toggle respected**: when **Use AOI = OFF**, filtering is disabled and overlays show **only a global total with per‑class breakdown**.
* **Per‑class filters**: choose which classes to count.
* **Advanced presets** (Fast/Balanced/Ultra) + expert panel (tiling, NMS/WBF, seam de‑dup, etc.).
* **Strict confidence**: detections must be **strictly greater than** the chosen `conf` value.
* **Clean overlays**: boxes/labels or centroids, plus a **bottom‑right summary**:

  * With AOIs → `Total: 45   AOI1: 15 (car 10, bus 5)   AOI2: 30 (car 18, bus 12)`
  * Without AOIs → `Total: 45 (car 28, bus 17)`
* **GIS‑friendly CSV export** for georeferenced images (points & boxes) with **non‑destructive filenames** (no overwrites).
* **Full detections CSV**: run‑level `detections_full.csv` with AOI names.
* **Immediate abort** with watchdog; label reliably flips to **Aborted.**

---

## 🖥️ Requirements

* **Windows 10/11** (primary target). macOS/Linux likely fine but untested.
* **Python 3.10–3.12** (tested with 3.12).
* **GPU** optional (CUDA auto‑used if available; CPU otherwise).

> **Models are not included.** Bring your own YOLO weights (`.pt`).

---

## 📦 Install (local, isolated)

```bash
python bootstrap_env.py
python start_app.py
```

This installs dependencies into `./pkgs` (or `./_pkgs`) and starts the app. Input and weights are **not prefilled** — click **Browse…** to choose them. Default output is `./output/`.

---

## 🚀 Workflow

1. **Browse… (Input)** → pick a folder or specific files.
2. **Browse… (Model)** → choose a YOLO `` file. Class list auto‑loads.
3. **Select classes** → at least one.
4. **(Optional) AOIs** → open **AOI Editor**, draw named polygons; saved to `input/aoi/*.json` and `input/aoi_masks/*.png`.
5. Choose **Fast / Balanced / Ultra** preset (or tweak **Advanced**).
6. **Start**. Use **Abort** to cancel immediately; label changes from *Aborting…* to *Aborted.* when the worker stops.

---

## 🧭 AOI behavior

* AOIs are persisted per image in `input/aoi/…` and `input/aoi_masks/…`.
* **Use AOI = ON**: detections must lie **in any AOI** (by centroid or box‑overlap). The overlay shows **Total + per‑AOI counts** (each with per‑class breakdown).
* **Use AOI = OFF**: AOIs (even if present on disk) are **ignored**; the overlay shows **only Total + global per‑class breakdown**.

**Editor hotkeys**

* Finish polygon: **Ctrl + Enter**
* Undo vertex: **Ctrl + Backspace**

---

## 🧪 Advanced (short)

* **Tiling**: `tile` (px), `overlap` (0–1)
* **Thresholds**: strict `conf` (kept if `conf_raw > conf`), `iou_nms`
* **WBF**: `use_wbf`, `wbf_iou` (auto if empty), `wbf_alpha`
* **Seam de‑dup**: `seam_band_factor`, `seam_weight`
* **Overlay**: `boxes` / `boxes_conf` / `centroid` and optional centroid dot

---

## 🧾 Outputs

All in the chosen **Output** folder (default `./output/`). Files get numeric suffixes to avoid overwrites.

* `annotated/<image>_annotated.jpg` — detections + bottom‑right summary (conditional AOI logic above).
* `results_per_image.csv` — per‑image class counts.
* `results_totals.json` — run totals.
* `detections_full.csv` — kept detections (with AOI name when AOI is ON).
* `gis/<image>__detections_p.csv` & `gis/<image>__detections_b.csv` — world‑coords for points/boxes (only if the image is georeferenced).

---

## 🔧 Troubleshooting

* **No detections** → verify classes, `conf` not too high, and model fits the data.
* **Overlay vs CSV mismatch** → CSV uses raw floats and strict `>`; overlay is rounded.

---

## 📚 Project layout

```
start_app.py
legacy_pt_runner.py
app_core.py
engine_loader.py
ui_panels.py, ui_advanced.py, widgets.py
geo_export.py
bootstrap_env.py
pkgs/  weights/  input/  output/
```

---

## 🗺️ Roadmap (next release)

**Features**

* Bring back **ONNX** runtime support for `.onnx` models.
* Optional: AOI summary order by **count** (desc) instead of name.
* Atomic file writes (tmp → rename) to avoid partial artifacts on abort.
* Config file for defaults (YAML), CLI/batch mode.
* Unit tests for AOI math and GIS exporters.
* take image / connect camera for analys here & now
* ios compatibilitet test
* segmentation model support


**Refactors / cleanup**

* Extract AOI helpers to `aoi_utils.py` (normalize polygons, masks, union).
* Move `unique_path` & friends to `utils.py` and reuse everywhere.
* Centralize CSV/JSON writers in `app_core.py` to remove duplication.
* Unify device/engine selection in `engine_loader.py` and pass to runners.
* Optional `overlay.py` to consolidate drawing/summary formatting.
* Lightweight logger helper for consistent formatting/levels.
* Unify progress reporting (ETA + smoothing) via one helper.

> The goal is to keep `start_app.py` thinner by moving utilities into small modules without changing current behavior.
