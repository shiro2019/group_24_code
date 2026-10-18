"""CPU-only tests of freeze safety and split leakage checks; no model downloads."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from roles.c import workflow as w
from src import common


class FreezeTests(unittest.TestCase):
    def setUp(self):
        test_root = common.REPO_ROOT / "outputs" / "test_tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=test_root)
        assert Path(self.tmp.name).resolve().is_relative_to(test_root.resolve())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.splits = self.root / "data/splits"
        self.splits.mkdir(parents=True)
        self.frozen = self.root / "configs/c_frozen"
        for obj, name, value in ((w, "REPO_ROOT", self.root), (w, "SPLITS_DIR", self.splits),
                                 (w, "FROZEN", self.frozen), (common, "REPO_ROOT", self.root),
                                 (common, "SPLITS_DIR", self.splits)):
            p = patch.object(obj, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.rows = {"tune": self.prompts(0, 500), "eval": self.prompts(500, 2000),
                     "diversity": self.prompts(500, 100)}
        self.save_splits()
        p = {"method": "sdxl_turbo", "split": "tune", "experiment": "e2",
             "name": "steps4_cfg0", "params": {"num_inference_steps": 4, "guidance_scale": 0.0},
             "seeds": [0], "models": {"stabilityai/sdxl-turbo": "pinned"}, "resolution": 512}
        self.selection = {"quality": p, "lowest_step": copy.deepcopy(p), "clip_tolerance": 1.0}
        self.selection["lowest_step"]["name"] = "steps1_cfg0"
        self.selection["lowest_step"]["params"]["num_inference_steps"] = 1
        self.selection_path = self.root / "selection.json"

    @staticmethod
    def prompts(start, count):
        return [{"prompt_id": f"coco{i}", "image_id": i, "prompt": f"prompt {i}"}
                for i in range(start, start + count)]

    def save_splits(self):
        for split, rows in self.rows.items():
            (self.splits / f"{split}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def freeze(self):
        self.selection_path.write_text(json.dumps(self.selection), encoding="utf-8")
        w.freeze(self.selection_path)

    def test_split_overlap_rejected(self):
        self.rows["eval"][0] = self.rows["tune"][0]
        self.save_splits()
        with self.assertRaisesRegex(ValueError, "overlap"):
            w.audit_splits()

    def test_wrong_split_size_rejected(self):
        self.rows["tune"].pop()
        self.save_splits()
        with self.assertRaisesRegex(ValueError, "500"):
            w.audit_splits()

    def test_freeze_preserves_source_and_derives_expected_counts(self):
        before = copy.deepcopy(self.selection)
        self.freeze()
        self.assertEqual(before, self.selection)
        manifest = json.loads((self.frozen / "manifest_sdxl_turbo.json").read_text())
        self.assertEqual(len(manifest["e1_configs"]), 2)
        canonical = yaml.safe_load((self.root / "configs/e1/sdxl_turbo.yaml").read_text())
        self.assertEqual(canonical["grid"], {"num_inference_steps": [1, 4]})
        self.assertTrue((self.root / "configs/e1/selection_sdxl_turbo.json").exists())
        cfg = yaml.safe_load((self.root / manifest["diversity_config"]).read_text())
        self.assertEqual(cfg["seeds"], list(range(8)))
        self.assertEqual(cfg["split"], "diversity")
        self.assertEqual(len(common.read_prompts(cfg)) * len(cfg["seeds"]), 800)
        self.freeze()  # idempotent
        self.selection["quality"]["params"]["num_inference_steps"] = 3
        with self.assertRaisesRegex(ValueError, "frozen file differs"):
            self.freeze()

    def test_coincident_quality_and_lowest_deduplicated(self):
        self.selection["lowest_step"] = copy.deepcopy(self.selection["quality"])
        self.freeze()
        manifest = json.loads((self.frozen / "manifest_sdxl_turbo.json").read_text())
        self.assertEqual(len(manifest["e1_configs"]), 1)

    def test_exploratory_and_limited_selections_rejected(self):
        for key, value in (("exploratory", True), ("limit", 10), ("split", "eval")):
            old = copy.deepcopy(self.selection)
            self.selection["quality"][key] = value
            with self.assertRaises(ValueError):
                self.freeze()
            self.selection = old

    def test_postfreeze_split_change_blocks_final_before_gpu(self):
        self.freeze()
        self.rows["tune"][0]["prompt"] = "different text"
        self.save_splits()
        with patch.object(w, "run") as run:
            with self.assertRaisesRegex(ValueError, "changed after freezing"):
                w.final("sdxl_turbo", "diversity")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
