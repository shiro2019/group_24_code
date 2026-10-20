"""Checks for the paired bootstrap intervals."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import bootstrap_ci as bc


def write_scores(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(1)
        self.base = [{"prompt_id": f"p{i}", "seed": 0, "clip_score": float(c), "image_reward": float(r)}
                     for i, (c, r) in enumerate(zip(rng.normal(26, 3, 400), rng.normal(0.5, 1, 400)))]

    def run_ci(self, other_rows, ref_rows=None, n_boot=2000):
        ref = write_scores(self.root / "ref.jsonl", ref_rows or self.base)
        other = write_scores(self.root / "other.jsonl", other_rows)
        counts, sums = bc.align([("ref", bc.load_scores(ref)), ("other", bc.load_scores(other))])
        rows = bc.bootstrap(counts, sums, "ref", n_boot=n_boot)
        return {(r["label"], r["metric"]): r for r in rows}

    def test_constant_shift_has_zero_width_paired_interval(self):
        shifted = [{**r, "image_reward": r["image_reward"] + 0.1} for r in self.base]
        row = self.run_ci(list(reversed(shifted)))[("other", "image_reward")]
        self.assertAlmostEqual(row["diff"], 0.1, places=9)
        self.assertAlmostEqual(row["diff_low"], 0.1, places=9)
        self.assertAlmostEqual(row["diff_high"], 0.1, places=9)

    def test_interval_width_matches_standard_error(self):
        row = self.run_ci(self.base)[("ref", "image_reward")]
        values = np.array([r["image_reward"] for r in self.base])
        expected = 2 * 1.96 * values.std() / np.sqrt(len(values))
        self.assertAlmostEqual(row["mean"], values.mean(), places=9)
        self.assertLess(abs((row["ci_high"] - row["ci_low"]) / expected - 1), 0.15)
        self.assertNotIn("diff", row)

    def test_seeds_of_one_prompt_are_resampled_together(self):
        # Two identical seeds per prompt add no information, so the interval must match the
        # one-seed interval instead of shrinking by sqrt(2).
        doubled = [{**r, "seed": s} for r in self.base for s in (0, 1)]
        single = self.run_ci(self.base)[("ref", "clip_score")]
        double = self.run_ci(doubled, ref_rows=doubled)[("ref", "clip_score")]
        self.assertEqual((double["prompts"], double["images"]), (400, 800))
        self.assertAlmostEqual(double["ci_low"], single["ci_low"], places=9)
        self.assertAlmostEqual(double["ci_high"], single["ci_high"], places=9)

    def test_metric_missing_in_one_input_is_skipped(self):
        clip_only = [{k: v for k, v in r.items() if k != "image_reward"} for r in self.base]
        with contextlib.redirect_stderr(io.StringIO()) as err:
            rows = self.run_ci(clip_only)
        self.assertEqual({m for _, m in rows}, {"clip_score"})
        self.assertIn("image_reward", err.getvalue())
        self.assertIn("diff", rows[("other", "clip_score")])

    def test_different_image_sets_are_refused(self):
        with self.assertRaises(ValueError):
            self.run_ci(self.base[:-1])

    def test_duplicate_and_empty_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            bc.load_scores(write_scores(self.root / "dup.jsonl", self.base + self.base[:1]))
        with self.assertRaises(ValueError):
            bc.load_scores(write_scores(self.root / "empty.jsonl", []))

    def test_main_writes_csv_and_refuses_duplicate_labels(self):
        (self.root / "cfg").mkdir()
        ref = write_scores(self.root / "cfg" / "per_image_scores.jsonl", self.base)
        out = self.root / "ci.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            bc.main(["--ref", str(ref), f"other={ref}", "--n-boot", "200", "--out", str(out)])
        with open(out, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(r["label"], r["metric"]) for r in rows],
                         [("cfg", "clip_score"), ("cfg", "image_reward"),
                          ("other", "clip_score"), ("other", "image_reward")])
        self.assertEqual(float(rows[2]["diff"]), 0.0)
        with self.assertRaises(SystemExit):
            bc.main(["--ref", f"x={ref}", f"x={ref}"])

    def test_label_parsing(self):
        label, path = bc.parse_input("outputs/e1/eval/sdxl_base/steps50_cfg7.5/per_image_scores.jsonl")
        self.assertEqual(label, "steps50_cfg7.5")
        self.assertEqual(bc.parse_input("base50=x.jsonl"), ("base50", Path("x.jsonl")))
        self.assertEqual(bc.parse_input("out/a=b/s.jsonl"), ("a=b", Path("out/a=b/s.jsonl")))


if __name__ == "__main__":
    unittest.main()
