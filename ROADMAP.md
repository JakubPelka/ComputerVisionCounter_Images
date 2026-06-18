# Roadmap

## v0.1.0 - cleanup release candidate

Goal: publish a clean baseline release without changing the working application logic.

- [x] Clean public README.
- [x] MIT license.
- [x] Expanded `.gitignore`.
- [x] Basic repository documentation.
- [x] Code moved to `src/` layout with MVC architecture.
- [x] Manual user smoke test on Windows after src refactor.
- [ ] Optional fresh-clone smoke test before GitHub release.
- [ ] Create GitHub release after testing.

## v0.1.x - safe maintenance

- Improve release packaging notes.
- Verify fresh setup on a clean Windows machine.
- Add a small non-private sample or documented sample-data pattern.
- Add screenshots to README if available.

## v1.x - further improvements

- Add automated tests for AOI logic and result exports.
- Split large modules only when the current behavior is protected by tests.

## Future ideas

- Repair and document ONNX support.
- Investigate segmentation model support.
- Improve batch/release packaging.
- Consider CLI mode for repeatable counting jobs.
