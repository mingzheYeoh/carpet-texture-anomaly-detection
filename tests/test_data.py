"""Synthetic files exercise split isolation and failed integrity checks."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from carpet_ad import data


class DataCheck(unittest.TestCase):
    def test_manifest_is_repeatable_and_rejects_leakage_and_missing_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "carpet"
            for directory in ("train/good", "test/good", "test/cut", "ground_truth/cut"):
                (root / directory).mkdir(parents=True)
            for index in range(10):
                Image.new("RGB", (8, 8), (index, 20, 30)).save(root / f"train/good/{index:03}.png")
            Image.new("RGB", (8, 8), (40, 50, 60)).save(root / "test/good/000.png")
            Image.new("RGB", (8, 8), (70, 80, 90)).save(root / "test/cut/000.bmp")
            mask = root / "ground_truth/cut/000_mask.png"
            Image.fromarray(np.eye(8, dtype=np.uint8) * 255).save(mask)
            manifest, audit = base / "split.csv", base / "audit.md"
            data.prepare_data(root, manifest, audit, seed=42)
            original = manifest.read_bytes()
            rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
            self.assertEqual(len(rows), 12)
            self.assertEqual({r["path"] for r in rows if r["split"] == "calibration"},
                             {"train/good/001.png", "train/good/008.png"})
            self.assertEqual(sum(r["split"] == "fit" for r in rows), 8)
            self.assertTrue(all(r["label"] == "0" for r in rows if r["split"] != "test"))
            self.assertEqual(next(r["mask_path"] for r in rows if r["label"] == "1"),
                             "ground_truth/cut/000_mask.png")
            data.prepare_data(root, manifest, audit, seed=42)
            self.assertEqual(manifest.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "existing manifest"):
                data.prepare_data(root, manifest, audit, seed=43)
            Image.new("L", (4, 4), 255).save(mask)
            with self.assertRaisesRegex(ValueError, "Mask dimensions"):
                data.prepare_data(root, manifest, audit, seed=42)
            self.assertEqual(manifest.read_bytes(), original)
            mask.unlink()
            with self.assertRaisesRegex(ValueError, "audit failed"):
                data.prepare_data(root, manifest, audit, seed=42)
            self.assertIn("Missing or ambiguous mask", audit.read_text(encoding="utf-8"))
            self.assertEqual(manifest.read_bytes(), original)
            Image.fromarray(np.eye(8, dtype=np.uint8) * 255).save(mask)
            (root / "test/good/000.png").write_bytes((root / "train/good/000.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "audit failed"):
                data.prepare_data(root, manifest, audit, seed=42)
            self.assertIn("Cross-split duplicate", audit.read_text(encoding="utf-8"))
            (root / "test/good/000.png").write_bytes(b"broken image")
            with self.assertRaisesRegex(ValueError, "audit failed"):
                data.prepare_data(root, manifest, audit, seed=42)
            self.assertIn("cannot identify image", audit.read_text(encoding="utf-8"))

    def test_missing_dataset_never_creates_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            with self.assertRaisesRegex(ValueError, "audit failed"):
                data.prepare_data(base / "missing", base / "split.csv", base / "audit.md")
            self.assertFalse((base / "split.csv").exists())


if __name__ == "__main__":
    unittest.main()
