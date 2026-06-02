# Repository review

This note captures the current understanding of the project before source-code refactoring. It is intentionally documentation-only.

## Functional summary

ComputerVision Counter Images is a local Windows desktop app for counting objects in still images with YOLO models. The main user flow is:

1. Select an image folder, individual image files, or capture a camera snapshot.
2. Select a YOLO `.pt` model.
3. Load/select model classes.
4. Optionally draw or reuse named AOI polygons.
5. Choose quality/preset settings and overlay style.
6. Run tiled detection.
7. Export annotated preview images plus CSV/JSON results.
8. Export GIS-friendly CSV layers when image georeferencing is available.

The stable path is Ultralytics YOLO `.pt`. ONNX handling exists, but should remain experimental until it has dedicated tests and clearer user-facing behavior.

## Current code map

```text
start.bat              Windows launcher; creates local working folders.
project_paths.py       Central project paths and local package path helper.
bootstrap_env.py       Local dependency bootstrap into ./_pkgs.
start_app.py           Tkinter app, workflow state, threading, runner routing.
ui_panels.py           Main UI layout helpers.
ui_advanced.py         Advanced settings and preset import/export UI.
widgets.py             Scrollable frame and AOI editor widget.
legacy_pt_runner.py    Primary .pt inference path: tiling, NMS, AOI filters, outputs.
app_core.py            Alternative engine-oriented core with config, WBF and batch API.
engine_loader.py       Device/model loading helpers plus ONNX metadata patching.
onnx_ultra_patch.py    Standalone ONNX metadata patch utility.
geo_export.py          GIS CSV export from worldfiles / raster metadata.
presets/               Tracked example preset.
input/ models/ output/ Local working folders; contents ignored by Git.
```

## Structure cleanup started

- Moved old Quick Start `.docx` / `.pdf` files into `docs/legacy/`.
- Moved cleanup overlay notes into `docs/legacy/`.
- Moved migration and broader roadmap notes into `docs/planning/`.
- Kept all Python source files in the repository root for now.
- Did not change application logic.

## Main improvement opportunities

1. Add behavior protection before refactoring: smoke test checklist, AOI unit tests, export tests, and a tiny non-private sample fixture.
2. Continue centralizing path handling around `project_paths.py`, including any remaining output and AOI artifact paths.
3. Consolidate AOI parsing/persistence/filtering, which currently appears in several modules.
4. Decide the runner boundary: either make `legacy_pt_runner.py` the official first runner interface, or merge it behind a cleaner engine abstraction with `app_core.py`.
5. Replace broad silent `except Exception/pass` paths with explicit logging or user-facing warnings where behavior can silently degrade.
6. Standardize run outputs: annotated images, full detections, per-image summary, totals, GIS layers, metadata and used preset.
7. Keep ONNX behind an experimental label until class metadata, device/runtime requirements and output parity are tested.
8. Add release packaging notes that separate source releases from optional app ZIPs and local `_pkgs` folders.
9. Consider a later `src/` migration only after paths and launcher behavior are covered by tests.

## Suggested next step

The next source-code step should stay small and protective: consolidate AOI helpers, then run a manual smoke test with AOI off and on. Avoid moving all `.py` files into `src/` until startup, path resolution and output generation are verified.
