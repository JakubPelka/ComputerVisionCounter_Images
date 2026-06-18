# Quickstart

## 1. Prepare folders

Create local folders if they do not already exist:

```text
input/
models/
output/
```

These folders are ignored by Git to avoid committing private images, model weights and generated results.

## 2. Add files locally

- Put your images in `input/` or choose another folder from the app.
- Put your YOLO `.pt` model in `models/` or choose another model path from the app.

Do not commit private images or model weights.

## 3. Start the app

Recommended on Windows:

```bat
start.bat
```

Manual start:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\start_app.py
```

The launcher installs only the minimal direct dependencies for the stable
YOLO `.pt` workflow into `.venv`. ONNX, ONNX Runtime and segmentation-specific
packages are intentionally not installed by default in this release candidate.

## 4. Run a counting job

1. Select input image folder or image files.
2. Select a YOLO `.pt` model.
3. Select classes.
4. Optionally draw AOIs.
5. Start processing.
6. Check `output/` for annotated images and result files.

## 5. Before release

Run a basic smoke test:

- fresh clone or clean local copy;
- empty `input/`, `models/`, `output/` folders;
- one known test image;
- one known `.pt` model;
- run with AOI off;
- run with AOI on;
- verify annotated output and CSV/JSON exports.
