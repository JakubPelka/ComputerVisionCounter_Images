# Known challenges

## 1. Validating the `src/` layout

The application code has already been moved into `src/` as part of the v0.2.0 release candidate. The remaining risk is not the move itself, but validating startup, imports, paths and output behavior after the refactor.

Recommended future approach:

- verify start from start.bat;
- verify manual start from .venv;
- verify input/models/output paths;
- verify AOI and export behavior;
- keep future structural changes small and tested.

## 2. ONNX support

ONNX support should be treated as experimental until tested. The stable user-facing path is YOLO `.pt` models. Generic ONNX files may miss metadata expected by Ultralytics-based loading.

The release-candidate bootstrap intentionally does not install ONNX or ONNX Runtime packages. Add them back only when the ONNX path has a tested workflow and clear dependency policy.

## 3. Segmentation models

Segmentation model support is a separate feature track. It should not be mixed with the cleanup release.

Segmentation-specific dependencies should stay out of the default bootstrap until the feature is designed and tested.

## 4. Release packaging

The public repository should not become a backup or distribution dump. Keep release assets intentional:

- source release through GitHub tags/releases;
- optional application ZIP only when tested;
- no private input data;
- no model weights unless there is a clear license decision;
- no `_pkgs/`, outputs or temporary archives in Git.

## 5. Testing before v0.2.0

Minimum smoke test before release:

- start app using `start.bat`;
- run detection with AOI off;
- run detection with AOI on;
- verify annotated output;
- verify CSV/JSON output;
- confirm no private/heavy files appear in GitHub Desktop before commit.

## 6. AOI consolidation

AOI JSON parsing has started moving into `aoi_utils.py`. The next safe step is to keep reducing duplicate AOI geometry logic while preserving the existing `polygon` / `points` / `pts` compatibility.
