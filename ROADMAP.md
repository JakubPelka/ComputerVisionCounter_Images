# Roadmap

## v0.1.0 - cleanup release candidate

Goal: publish a clean baseline release without changing the working application logic.

- [x] Clean public README.
- [x] MIT license.
- [x] Expanded `.gitignore`.
- [x] Basic repository documentation.
- [ ] Manual smoke test on Windows.
- [ ] Create GitHub release after testing.

## v0.1.x - safe maintenance

- Improve release packaging notes.
- Verify fresh setup on a clean Windows machine.
- Add a small non-private sample or documented sample-data pattern.
- Add screenshots to README if available.

## v1.x - structural refactor

- Move working scripts into `src/` only after path handling is reviewed and tested.
- Add automated tests for AOI logic and result exports.
- Centralize path handling for `input/`, `models/`, `output/`, `_pkgs/`.
- Split large modules only when the current behavior is protected by tests.

## Future ideas

- Repair and document ONNX support.
- Investigate segmentation model support.
- Improve batch/release packaging.
- Consider CLI mode for repeatable counting jobs.
