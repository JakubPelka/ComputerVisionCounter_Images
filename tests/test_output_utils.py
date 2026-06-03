from __future__ import annotations

import unittest
from pathlib import Path

from output_utils import (
    DETECTIONS_FULL_CSV,
    GIS_AOIS_CSV_SUFFIX,
    GIS_DETECTIONS_BOX_CSV_SUFFIX,
    GIS_DETECTIONS_POINT_CSV_SUFFIX,
    RESULTS_PER_IMAGE_CSV,
    RESULTS_TOTALS_JSON,
    RUN_METADATA_JSON,
    class_names_from_counts,
    per_image_count_table,
    suffixed_csv_name,
    unique_path,
)


class OutputUtilsTest(unittest.TestCase):
    def test_output_file_names_are_stable(self):
        self.assertEqual(DETECTIONS_FULL_CSV, "detections_full.csv")
        self.assertEqual(RESULTS_PER_IMAGE_CSV, "results_per_image.csv")
        self.assertEqual(RESULTS_TOTALS_JSON, "results_totals.json")
        self.assertEqual(RUN_METADATA_JSON, "run_metadata.json")
        self.assertEqual(GIS_AOIS_CSV_SUFFIX, "__aois.csv")
        self.assertEqual(GIS_DETECTIONS_POINT_CSV_SUFFIX, "__detections_point.csv")
        self.assertEqual(GIS_DETECTIONS_BOX_CSV_SUFFIX, "__detections_box.csv")

    def test_suffixed_csv_name(self):
        self.assertEqual(suffixed_csv_name("image_01", GIS_AOIS_CSV_SUFFIX), "image_01__aois.csv")

    def test_per_image_count_table_sorts_classes_and_fills_missing(self):
        header, rows = per_image_count_table(
            [
                ("a.jpg", {"truck": 2}),
                ("b.jpg", {"car": 1, "truck": 3}),
            ]
        )
        self.assertEqual(header, ["image", "car", "truck"])
        self.assertEqual(rows, [["a.jpg", 0, 2], ["b.jpg", 1, 3]])

    def test_class_names_from_counts(self):
        self.assertEqual(
            class_names_from_counts([("a", {"z": 1, "a": 1}), ("b", {"m": 2})]),
            ["a", "m", "z"],
        )

    def test_unique_path_returns_same_path_when_missing(self):
        self.assertEqual(unique_path(Path("definitely_missing_file.csv")), Path("definitely_missing_file.csv"))


if __name__ == "__main__":
    unittest.main()
