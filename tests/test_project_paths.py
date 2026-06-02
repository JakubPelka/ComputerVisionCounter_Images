from __future__ import annotations

import sys
import unittest

import project_paths as paths


class ProjectPathsTest(unittest.TestCase):
    def test_project_root_contains_public_files(self):
        self.assertTrue((paths.PROJECT_ROOT / "README.md").is_file())
        self.assertTrue((paths.PROJECT_ROOT / "start_app.py").is_file())

    def test_working_dirs_are_under_project_root(self):
        for path in paths.WORKING_DIRS:
            self.assertTrue(path.is_absolute())
            self.assertTrue(path.is_relative_to(paths.PROJECT_ROOT))

    def test_named_dirs_match_expected_layout(self):
        self.assertEqual(paths.INPUT_DIR.name, "input")
        self.assertEqual(paths.MODELS_DIR.name, "models")
        self.assertEqual(paths.OUTPUT_DIR.name, "output")
        self.assertEqual(paths.PRESETS_DIR.name, "presets")

    def test_add_local_package_paths_is_reversible_for_tests(self):
        original = list(sys.path)
        try:
            added = paths.add_local_package_paths(strict=False)
            for path in added:
                self.assertIn(str(path), sys.path)
        finally:
            sys.path[:] = original


if __name__ == "__main__":
    unittest.main()
