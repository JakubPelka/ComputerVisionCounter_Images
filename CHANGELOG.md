# Changelog

## Unreleased

### Repository cleanup

- Added MIT license.
- Rewrote public README for a clearer open-source / release-candidate presentation.
- Added expanded `.gitignore` to block local packages, models, input data, outputs, logs, caches, archives and secrets.
- Added `.gitattributes` for more predictable line endings.
- Added repository documentation under `docs/`.
- Added placeholder README files for local-only folders: `input/`, `models/`, `sample_data/`, `presets/`.
- Identified known challenges before future development: `src/` refactor, ONNX support, segmentation models, release packaging and tests.
- Moved legacy and planning documents out of the repository root into `docs/legacy/` and `docs/planning/`.
- Added `docs/REPOSITORY_REVIEW.md` to capture functionality, current architecture and improvement opportunities before source refactoring.
- Added `project_paths.py` as a central place for repository paths and local package path handling.
- Added lightweight `unittest` coverage for project path behavior.
- Skipped unreadable local package folders when adding local dependency paths, avoiding hard `PermissionError` failures from inaccessible `_pkgs` folders.
- Hardened AOI widget and `.pt` runner imports so the app module can still load when optional OpenCV-backed paths are unavailable.
- Added `aoi_utils.py` for shared AOI JSON normalization across the UI and `.pt` runner paths.
- Added unit tests for AOI normalization compatibility with `polygon`, `points`, `pts`, plain point lists and mapping-style AOI inputs.
- Added `run_metadata.json` generation for the main legacy `.pt` runner path.
- Extended run metadata with input paths and optional runner/quality context.
- Added unit coverage for run metadata output.
- Added `output_utils.py` with shared output filenames, CSV/JSON writers and per-image count table helpers.
- Switched legacy `.pt` run-level CSV/JSON artifact writing to the shared output helpers while preserving unique output names.
- Switched GIS CSV export writing to the shared output helpers while preserving existing GIS filenames.
- Added unit coverage for GIS affine conversion, WKT formatting and GIS output filename suffixes.
- Aligned `requirements.txt` around the minimal direct dependency set for the stable YOLO `.pt` workflow.
- Kept ONNX, ONNX Runtime, segmentation-specific and GIS-heavy optional packages out of the default `.venv` path for the v0.1.0 release candidate.
- Added `.gitignore` exceptions for documentation screenshots under `docs/images/` while keeping local datasets and generated images ignored.
- Moved application code into `src/`.
- Added MVC-style separation with `ui_main.py`, `app_controller.py` and `runners/`.
- Replaced the old `bootstrap_env.py` workflow with a standard `.venv` startup through `start.bat`.
- Simplified `project_paths.py` to centralize repository folders without forcing `_pkgs` into runtime.
- Updated README and Quickstart for the new `src/` layout.
- User smoke test passed for standard app usage.

## v0.1.0 candidate

Initial cleaned release candidate after repository hygiene pass and structural refactor into `src/`.
