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

## v0.1.0 candidate

Initial cleaned release candidate after repository hygiene pass. The working application code is intentionally not refactored in this cleanup step.
