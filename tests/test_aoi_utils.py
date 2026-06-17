from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import unittest

from aoi_utils import normalize_aois, normalize_aois_as_tuples


class AoiUtilsTest(unittest.TestCase):
    def test_new_aoi_json_shape(self):
        data = {
            "image": "example.jpg",
            "aois": [
                {"name": "Zone A", "points": [[1, 2], [3, 4], [5, 6]]},
            ],
        }
        self.assertEqual(
            normalize_aois(data),
            [{"name": "Zone A", "polygon": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}],
        )

    def test_legacy_single_polygon_shape(self):
        data = {"points": [[1, 2], [3, 4], [5, 6]]}
        self.assertEqual(normalize_aois(data)[0]["name"], "AOI 1")

    def test_plain_point_list_ui_default(self):
        data = [[1, 2], [3, 4], [5, 6]]
        self.assertEqual(normalize_aois(data)[0]["name"], "AOI 1")

    def test_runner_tuple_defaults_can_match_legacy_runner(self):
        data = [[1, 2], [3, 4], [5, 6]]
        self.assertEqual(
            normalize_aois_as_tuples(data, point_list_name="AOI", indexed_item_names=True),
            [("AOI", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])],
        )

    def test_mapping_shape_uses_mapping_keys_as_names(self):
        data = {"North": [[1, 2], [3, 4], [5, 6]]}
        self.assertEqual(normalize_aois(data)[0]["name"], "North")

    def test_invalid_or_too_short_polygons_are_skipped(self):
        self.assertEqual(normalize_aois({"points": [[1, 2], [3, 4]]}), [])


if __name__ == "__main__":
    unittest.main()
