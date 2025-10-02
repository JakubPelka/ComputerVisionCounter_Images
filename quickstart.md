# ComputerVision Counter — Quick Start

This short guide reflects the current behavior: **AOI toggle respected**, **clean overlay summary**, and **immediate abort**.

---

## 1) Install once

```bash
python bootstrap_env.py
```

Installs dependencies into a local `./pkgs` folder (portable; no system pollution).

---

## 2) Launch

```bash
python start_app.py
```

Choose **Input** and **Model** via the GUI (not prefilled). Default output is `./output/`.

> **Supported models:** Ultralytics **`.pt`** only. **ONNX is disabled in this release** and will arrive in the next release.

---

## 3) Load data & model

1. **Browse… (Input)** → pick a folder or select files.
2. **Browse… (Model)** → choose a YOLO **`.pt`** file. Class names auto‑load.
3. **Select classes** (at least one).

**Get a test model (YOLO11n)**
We do **not** bundle third‑party weights. To try the app quickly, fetch a small official model from Ultralytics:

* Ultralytics Detect task: [https://docs.ultralytics.com/tasks/detect/](https://docs.ultralytics.com/tasks/detect/)
* YOLO11 models overview: [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)

> Tip: choose **YOLO11n** (nano) for the fastest test.

**More model sources**

* **Ultralytics Docs** (links above) — official `.pt` weights.
* **Ultralytics on Hugging Face** — search for `yolo11n.pt` and related variants.
* **Model zoos** (varied licenses): Roboflow Universe, Hugging Face Hub. Always verify the license of any third‑party model before use.

---

## 4) (Optional) Draw AOIs

Open **AOI Editor** and draw named polygons.

* Finish polygon: **Ctrl + Enter**
* Undo vertex: **Ctrl + Backspace**
* AOIs auto‑save to `input/aoi/*.json` + `input/aoi_masks/*.png`.

**AOI modes**

* **Centroid** (default)
* **Box overlap** with `aoi_box_frac` (e.g., 0.20 = 20%).

**Important:** The main **Use AOI** toggle is authoritative — if **OFF**, AOIs (even if present on disk) are **ignored**.

---

## 5) Choose quality

Use the **Fast / Balanced / Ultra** presets, or open **Advanced** for tiling, thresholds (strict `conf`), WBF, seam de‑dup, and overlay controls.

---

## 6) Run & abort

Click **Start**. Progress is tile‑based with ETA.

Click **Abort** to stop immediately; the label changes from **Aborting…** to **Aborted.** as soon as the worker finishes the current tile.

---

## 7) Outputs

All in `./output/` (non‑destructive filenames):

* `annotated/<image>_annotated.jpg`

  * **With AOIs** → `Total …   AOI1 …   AOI2 …`
  * **Without AOIs** → `Total … (per‑class breakdown)`
* `results_per_image.csv` & `results_totals.json`
* `detections_full.csv` (kept detections; includes AOI name when AOI is ON)
* `gis/<image>__detections_point.csv` and `gis/<image>__detections_box.csv` for georeferenced images

---

## 8) Tips

* Ensure **at least one class** is selected.
* Larger `tile` + higher `overlap` → better quality, slower.
* If overlays and CSV disagree slightly, remember CSV uses raw floats and strict `>` on `conf`.

---

## 🔭 Roadmap (next release)

* **ONNX runtime support** (first official release).
* **Segmentation models** support.
* Further improvements to AOI handling and overlay summaries.

---

### Licensing note

We don’t include third‑party model weights in the app package. Ultralytics YOLO models are provided under **AGPL‑3.0** (or a commercial **Enterprise License**). Review the model’s license before **redistributing** any weights or using them in commercial settings.
