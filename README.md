# ComputerVision Counter Images

**Status:** ACTIVE / pre-release cleanup  
**Planned release:** v0.2.0 candidate  
**License:** MIT  
**Primary platform:** Windows 10/11  

ComputerVision Counter Images is a small desktop application for counting objects in images with your own YOLO models. It is designed for users who want a practical no-code workflow: select images, select a model, choose classes, optionally draw Areas of Interest (AOIs), run detection, and export readable results.

The application runs locally. Images, model weights and outputs stay on your machine.

## What it does

- Counts objects in still images using YOLO `.pt` models.
- Lets the user select which model classes should be counted.
- Supports optional AOI filtering with named polygon zones.
- Creates annotated preview images.
- Exports CSV/JSON result files for further analysis.
- Provides GIS-friendly exports for georeferenced images when supported by the input data.
- Uses a local Python virtual environment so the project does not need to modify the global Python environment.

## Current scope

The stable path for this release is **Ultralytics YOLO `.pt` models**.

ONNX support is treated as experimental/known issue and should not be presented as the main supported workflow until it is fixed and tested. Detection models are not included in this repository. Model files can be large and may have separate licenses.

## Requirements

- Windows 10/11.
- Python 3.10-3.12. Python 3.12 has been used during development/testing.
- A YOLO `.pt` model compatible with Ultralytics.
- Optional NVIDIA GPU/CUDA for faster inference. CPU can work but may be slower.

Dependency policy for this release candidate: `start.bat` and
`requirements.txt` intentionally cover only the minimal direct dependencies for
the stable YOLO `.pt` workflow via a standard `.venv`. ONNX, ONNX Runtime and segmentation-specific
dependencies are not installed by default in this release candidate.

## Quick start

Clone or download the repository, place your model in a local `models/` folder, place test images in a local `input/` folder, then start the app.

```bat
start.bat
```

Alternative manual start:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\start_app.py
```

The app creates or uses local working folders such as `input/`, `models/`, and `output/`. These folders and `.venv` are intentionally ignored by Git except for small README placeholder files.

## Basic workflow

1. Start the application.
2. Choose an input folder or image files.
3. Choose a YOLO `.pt` model.
4. Select the classes you want to count.
5. Optionally draw AOIs.
6. Choose a quality/preset setting.
7. Run the counter.
8. Review annotated images and exported CSV/JSON files in `output/`.

## Repository layout

The codebase is modularized within the `src/` directory.

```text
ComputerVisionCounter_Images/
|-- README.md
|-- LICENSE
|-- CHANGELOG.md
|-- ROADMAP.md
|-- requirements.txt
|-- .gitignore
|-- .gitattributes
|-- start.bat
|-- start.sh
|-- src/
|   |-- start_app.py
|   |-- ui_main.py
|   |-- app_controller.py
|   |-- project_paths.py
|   |-- aoi_utils.py
|   |-- output_utils.py
|   |-- runners/
|   |   |-- base.py
|   |   |-- factory.py
|   |   |-- legacy_runner.py
|-- docs/
|   |-- QUICKSTART.md
|   |-- PROJECT_STRUCTURE.md
|   |-- KNOWN_CHALLENGES.md
|   |-- REPOSITORY_REVIEW.md
|   |-- RELEASE_NOTES_v0.2.0.md
|   |-- legacy/
|   `-- planning/
|-- presets/
|   |-- README.md
|   `-- cv_counter_preset.json
|-- input/
|   `-- README.md
|-- models/
|   `-- README.md
|-- output/
|   `-- .gitkeep
|-- sample_data/
|   `-- README.md
|-- tests/
|   |-- test_aoi_utils.py
|   |-- test_geo_export.py
|   |-- test_output_utils.py
|   |-- test_run_metadata.py
|   `-- test_project_paths.py
`-- tools/
    `-- README.md
```

## What should not be committed

Do not commit:

- model weights: `.pt`, `.onnx`, `.engine`, `.tflite`, `.pb`, `.h5`;
- local images or private datasets;
- output files, annotated images, logs and generated CSV/JSON results;
- `_pkgs/`, `pkgs/`, virtual environments, cache folders;
- local experiments, backups, old ZIP packages and temporary folders;
- secrets, tokens, `.env` files or private paths.

## License

The project code is released under the MIT License.

The MIT License allows use, modification, redistribution and commercial use of the code, as long as the copyright notice and license text are preserved. Detection models, datasets and third-party libraries may have separate licenses.

## Known challenges

See [`docs/KNOWN_CHALLENGES.md`](docs/KNOWN_CHALLENGES.md).
