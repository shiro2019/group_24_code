"""Checks for the paired bootstrap intervals."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import bootstrap_ci as bc


def write_scores(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(1)
        self.base = [{"prompt_id": f"p{i}", "seed": 0, "clip_score": float(c), "image_reward": float(r)}
                     for i, (c, r) in enumerate(zip(rng.normal(26, 3, 200), rng.normal(0.5, 1, 200)))]

    def run_ci(self, other_rows):
        write_scores(self.root / "ref.jsonl", self.base)
        write_scores(self.root / "other.jsonl", other_rows)
        inputs = [("ref", bc.load_scores(self.root / "ref.jsonl")),
                  ("other", bc.load_scores(self.root / "other.jsonl"))]
        rows = bc.bootstrap(bc.align(inputs), "ref", n_boot=2000)
        return {(r["label"], r["metric"]): r for r in rows}

    def test_constant_shift_has_zero_width_paired_interval(self):
        shifted = [{**r, "image_reward": r["image_reward"] + 0.1} for r in self.base]
        row = self.run_ci(list(reversed(shifted)))[("other", "image_reward")]
        self.assertAlmostEqual(row["diff"], 0.1, places=9)
        self.assertAlmostEqual(row["diff_low"], 0.1, places=9)
        self.assertAlmostEqual(row["diff_high"], 0.1, places=9)

    def test_interval_contains_mean(self):
        row = self.run_ci(self.base)[("ref", "clip_score")]
        self.assertLess(row["ci_low"], row["mean"])
        self.assertGreater(row["ci_high"], row["mean"])
        self.assertNotIn("diff", row)

    def test_different_image_sets_are_refused(self):
        with self.assertRaises(ValueError):
            self.run_ci(self.base[:-1])

    def test_duplicate_image_is_refused(self):
        write_scores(self.root / "dup.jsonl", self.base + self.base[:1])
        with self.assertRaises(ValueError):
            bc.load_scores(self.root / "dup.jsonl")

    def test_label_defaults_to_config_folder(self):
        label, path = bc.parse_input("outputs/e1/eval/sdxl_base/steps50_cfg7.5/per_image_scores.jsonl")
        self.assertEqual(label, "steps50_cfg7.5")
        self.assertEqual(bc.parse_input("base50=x.jsonl"), ("base50", Path("x.jsonl")))


if __name__ == "__main__":
    unittest.main()
