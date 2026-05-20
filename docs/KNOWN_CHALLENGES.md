# Known challenges

## 1. Move scripts to `src/`

The repository should eventually move application code from the root into `src/`. This should not be done blindly because path handling may change. The current release-candidate cleanup keeps code in place to avoid breaking a working app.

Recommended future approach:

- centralize project paths in one helper module;
- keep `input/`, `models/`, `output/` relative to the repository root, not relative to a moved script file;
- test start from `start.bat`, from terminal and from a fresh clone;
- only then move files to `src/`.

## 2. ONNX support

ONNX support should be treated as experimental until tested. The stable user-facing path is YOLO `.pt` models. Generic ONNX files may miss metadata expected by Ultralytics-based loading.

## 3. Segmentation models

Segmentation model support is a separate feature track. It should not be mixed with the cleanup release.

## 4. Release packaging

The public repository should not become a backup or distribution dump. Keep release assets intentional:

- source release through GitHub tags/releases;
- optional application ZIP only when tested;
- no private input data;
- no model weights unless there is a clear license decision;
- no `_pkgs/`, outputs or temporary archives in Git.

## 5. Testing before v0.1.0

Minimum smoke test before release:

- start app using `start.bat`;
- run detection with AOI off;
- run detection with AOI on;
- verify annotated output;
- verify CSV/JSON output;
- confirm no private/heavy files appear in GitHub Desktop before commit.
