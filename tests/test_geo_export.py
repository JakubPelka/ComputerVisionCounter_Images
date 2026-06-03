from __future__ import annotations

import unittest

from geo_export import _poly_wkt, pix2geo


class GeoExportTest(unittest.TestCase):
    def test_pix2geo_applies_affine_transform(self):
        aff = (2.0, 0.5, 100.0, -0.25, -3.0, 200.0)
        self.assertEqual(pix2geo(aff, 10.0, 4.0), (122.0, 185.5))

    def test_poly_wkt_uses_fixed_precision(self):
        self.assertEqual(
            _poly_wkt([(1, 2), (3.1234567, 4.5)]),
            "POLYGON((1.000000 2.000000, 3.123457 4.500000))",
        )


if __name__ == "__main__":
    unittest.main()
