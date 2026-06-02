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

## v0.1.0 candidate

Initial cleaned release candidate after repository hygiene pass. The working application code is intentionally not refactored in this cleanup step.
