# Project structure

This project has been cleanly refactored into an MVC architecture to ensure long-term stability and maintainability.

## Current release-candidate structure

```text
README.md                 public project description
LICENSE                   MIT license
CHANGELOG.md              change history
ROADMAP.md                planned development
requirements.txt          dependency reference
.gitignore                blocks generated/private/heavy files
.gitattributes            line-ending and binary rules
start.bat                 Windows launcher (creates .venv)
start.sh                  Linux launcher (creates .venv)
src/                      Main application source code
|-- start_app.py          Main entrypoint
|-- ui_main.py            UI layout and bindings (View)
|-- app_controller.py     Execution logic (Controller)
|-- project_paths.py      Central project paths reference
|-- aoi_utils.py          Shared AOI normalization helpers
|-- output_utils.py       Shared run output filenames
|-- runners/              Inference engine implementations
|   |-- base.py           Abstract BaseModelRunner
|   |-- factory.py        RunnerFactory
|   |-- legacy_runner.py  YOLO .pt runner implementation
docs/                     documentation
presets/                  tracked example presets
input/                    local input data, ignored except README
models/                   local model files, ignored except README
output/                   generated outputs, ignored
sample_data/              documented place for tiny non-private samples
tests/                    lightweight behavior-protection tests
```

## Documentation structure

Active user-facing and release-candidate docs live directly under `docs/`.
Historical files and old generated quick-start documents live under `docs/legacy/`.
Planning notes and migration checklists live under `docs/planning/`.

## Architecture Note

The application separates concerns cleanly:
- `ui_main.py` handles Tkinter layouts and forms.
- `app_controller.py` manages threading, orchestrates model execution, and handles IO.
- `runners/` directory allows easily swapping inference engines (e.g. YOLO vs ONNX) without altering core logic.
