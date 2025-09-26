# ComputerVision Counter — Quick Start

This guide shows how to get from zero to counting in minutes, using the same flow described in the README and product page.

---

## 1) What you need

* **Windows 10/11** (officially tested)
* **Python 3.10–3.12** installed
* **Your own YOLO object‑detection model** (models are **not** included)

  * Recommended: Ultralytics **`.pt`** weights
  * **ONNX** is available but **experimental**
* **Images** to process (JPG/PNG/TIFF)
* **GPU** optional — used automatically if available; otherwise CPU

> All processing runs **locally** inside the app folder. No data leaves your machine.

---

## 2) Install (one‑time)

From the project folder run:

```bash
python bootstrap_env.py
```

This installs all dependencies into a local **`pkgs/`** folder so nothing touches your global Python.

---

## 3) Launch the app

```bash
python start_app.py
```

The app opens with a clean GUI. By default, results will be written to **`./output/`** (created if missing).

> Input and weights are **not prefilled** — click **Browse…** to choose them.

---

## 4) Load data & model (3 clicks)

1. **Browse… (Input)** → pick a folder **or** specific files.
2. **Browse… (Model)** → select your **`.pt`** (recommended) or **`.onnx`** (experimental) weights.

   * The class list auto‑loads from your model.
3. **Pick classes** → at least one must be selected.

> Tip: Use **batch mode** by selecting a folder with many images.

---

## 5) (Optional) Define AOIs

Click **AOI Editor** to draw named zones per image:

* **Finish polygon**: **Ctrl + Enter**
* **Undo last vertex**: **Ctrl + Backspace**
* AOIs are **auto‑saved** per image to `input/aoi/<image>.json` and `input/aoi_masks/<image>.png`.
* Reopen the editor anytime to **import/export** AOIs and verify visually.
* AOI modes in main UI:

  * **Centroid** (default): a detection counts if its **box center** lies inside an AOI.
  * **Box overlap**: counts if AOI overlap area ≥ **`aoi_box_frac`** (e.g., `0.20` = 20%).

---

## 6) Choose a preset (or tune Advanced)

* **Fast / Balanced / Ultra** presets control tile size, overlap, and thresholds.
* Click **Advanced** to adjust:

  * **Tiling**: `tile` (px), `overlap` (0–1)
  * **Thresholds**: strict `conf` (detections must be **>`conf`**), `iou_nms`
  * **WBF**: `use_wbf`, `wbf_iou` (auto suggested), `wbf_alpha`
  * **Seam de‑dup** (anti‑tile‑edge duplicates): `seam_band_factor`, `seam_weight`
  * **Overlay**: boxes/labels vs centroids

> The **Apply anti‑seam de‑dup** button sets sensible values; the window stays open so you can compare.

---

## 7) Run

Press **Start** (or **Ctrl + Enter**). The progress bar advances per tile and shows ETA.
You can **Abort** any time.

---

## 8) Outputs (in `./output/`)

* **Annotated previews**: `annotated/<image>_ann.jpg` with boxes/labels or centroids **and** a bottom‑right per‑class total.
* **Per‑image summary**: `results_per_image.csv`
* **Run totals**: `results_totals.json`
* **Full detections** (kept after conf/AOI): `detections_full.csv` with columns:
  `image, cls, conf, x1, y1, x2, y2, cx, cy, in_aoi`
* **GIS‑friendly CSVs** (if image is georeferenced):

  * `<image>_3857__detections_p.csv` (points)
  * `<image>_3857__detections_b.csv` (boxes)

> The app avoids overwriting by adding numeric suffixes to output files.

---

## 9) Tips & gotchas

* **Select at least one class** before starting.
* **Centroid** mode is best for discrete objects; use **box overlap** for large/irregular ones.
* Larger **tile** + higher **overlap** can improve quality but takes longer.
* Confidence labels on images are **rounded**; the filter uses **raw floats** and **strict greater‑than**.

---

## 10) Troubleshooting

* **“Low‑conf detections pass my threshold”** → The filter is strict `> conf` on raw floats. Check `detections_full.csv` values.
* **Geo exports don’t appear in GIS** → Use the **CSV** outputs (points/boxes). They’re the most compatible.
* **ONNX errors (stride/shape/NMS)** → Try the original **`.pt`** model. ONNX support is improving.
* **No detections** → Verify classes selected, `conf` not too high, and your model fits the data.

---

## FAQ (short)

* **macOS/Linux?** Officially tested on Windows; other OS may require manual setup.
* **Are models included?** No — bring your own weights.
* **GPU required?** No — used automatically if available.
* **Offline?** Yes — fully local.

---

**Happy counting!** 🎯
