from __future__ import annotations

import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from app_core import InferConfig, build_run_metadata


class RunMetadataTest(unittest.TestCase):
    def test_build_run_metadata_includes_inputs_params_totals_and_extra(self):
        outdir = Path("output")
        inputs = [Path("input/a.jpg"), Path("input/b.jpg")]
        cfg = InferConfig(model_path="models/example.pt", conf=0.42, classes=[0, 2])
        totals = {"car": 3}

        data = build_run_metadata(
            outdir,
            inputs,
            cfg,
            totals,
            extra={"runner": "legacy_pt_runner", "quality": "Fast"},
        )

        self.assertEqual(data["inputs_count"], 2)
        self.assertEqual(data["inputs"], [str(p) for p in inputs])
        self.assertEqual(data["params"]["model_path"], "models/example.pt")
        self.assertEqual(data["params"]["conf"], 0.42)
        self.assertEqual(data["params"]["classes"], [0, 2])
        self.assertEqual(data["totals"], totals)
        self.assertEqual(data["runner"], "legacy_pt_runner")
        self.assertEqual(data["quality"], "Fast")


if __name__ == "__main__":
    unittest.main()
