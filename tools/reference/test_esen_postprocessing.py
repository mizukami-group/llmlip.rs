"""Synthetic tests: no upstream checkpoint or reference framework required."""
import math
import unittest
from export_esen_oam_state import extract_postprocessing


class PostprocessingTests(unittest.TestCase):
    def checkpoint(self):
        return {"normalizers": {"energy": {"mean": 0.0, "rmsd": 2.0}},
                "elementrefs": {"energy": {"element_references": [0.25] * 119}}}

    def test_synthetic_values_preserved(self):
        result = extract_postprocessing(self.checkpoint())
        self.assertEqual(result["energy_rmsd"], 2.0)
        self.assertEqual(result["element_references"], [0.25] * 119)

    def test_missing_checkpoint_data_rejected(self):
        with self.assertRaises(ValueError):
            extract_postprocessing({})

    def test_invalid_scales_and_mean_rejected(self):
        for scale in (0, -1, math.nan, math.inf):
            checkpoint = self.checkpoint()
            checkpoint["normalizers"]["energy"]["rmsd"] = scale
            with self.assertRaises(ValueError):
                extract_postprocessing(checkpoint)
        checkpoint = self.checkpoint()
        checkpoint["normalizers"]["energy"]["mean"] = 1
        with self.assertRaises(ValueError):
            extract_postprocessing(checkpoint)

    def test_invalid_references_rejected(self):
        for values in ([0.0], [math.nan] * 119, [math.inf] * 119):
            checkpoint = self.checkpoint()
            checkpoint["elementrefs"]["energy"]["element_references"] = values
            with self.assertRaises(ValueError):
                extract_postprocessing(checkpoint)


if __name__ == "__main__":
    unittest.main()
