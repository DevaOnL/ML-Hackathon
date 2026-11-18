"""Small synthetic tests for immutable report-image crops; no OCR/browser required."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("report_images", ROOT / "scripts/prepare_report_images.py")
images = module_from_spec(spec)
spec.loader.exec_module(images)


class TestReportImages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.png"
        image = Image.new("RGB", (8, 6))
        image.putdata([(i, i * 2, i * 3) for i in range(48)])
        image.save(self.source)
        self.digest = images.sha(self.source)
        self.out = self.root / "crops"

    def test_exact_original_pixels_and_deterministic_outputs(self):
        box = (1, 2, 7, 5)
        first, second = self.out / "a.png", self.out / "b.png"
        result = images.crop_image(self.source, self.digest, box, first, self.out)
        images.crop_image(self.source, self.digest, box, second, self.out)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with Image.open(self.source) as original, Image.open(first) as crop:
            self.assertEqual(original.crop(box).tobytes(), crop.tobytes())
            self.assertEqual(crop.size, (6, 3))
        self.assertTrue(result["pixel_equality_checks"])

    def test_wrong_source_hash_rejected_before_writing(self):
        with self.assertRaisesRegex(ValueError, "source hash"):
            images.crop_image(self.source, "0" * 64, (0, 0, 4, 4), self.out / "x.png", self.out)
        self.assertFalse(self.out.exists())

    def test_invalid_boxes_rejected(self):
        for box in ((-1, 0, 3, 3), (0, 0, 9, 3), (2, 2, 2, 3), (0, 3, 4, 2), (0, 0, 4.0, 3)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                images.crop_image(self.source, self.digest, box, self.out / "x.png", self.out)
        self.assertFalse(self.out.exists())

    def test_existing_output_refused_and_unchanged(self):
        target = self.out / "x.png"
        images.crop_image(self.source, self.digest, (0, 0, 4, 4), target, self.out)
        before = target.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            images.crop_image(self.source, self.digest, (1, 1, 4, 4), target, self.out)
        self.assertEqual(before, target.read_bytes())

    def test_outside_or_symlink_output_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            images.crop_image(self.source, self.digest, (0, 0, 4, 4), self.root / "bad.png", self.out)
        self.out.mkdir()
        (self.out / "link").symlink_to(self.out, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            images.crop_image(self.source, self.digest, (0, 0, 4, 4), self.out / "link" / "bad.png", self.out)

    def test_create_refuses_existing_manifest_before_cropping(self):
        self.out.mkdir()
        manifest = self.out / "manifest.json"
        manifest.write_text("immutable")
        with patch.object(images, "OUT", self.out), patch.object(images, "MANIFEST", manifest), \
             patch.object(images, "sources_from_evidence", return_value={}):
            with self.assertRaisesRegex(ValueError, "overwrite"):
                images.create()
        self.assertEqual(manifest.read_text(), "immutable")

    def test_fixed_report_contract_has_exactly_twelve_readable_crops(self):
        self.assertEqual(len(images.SPECS), 12)
        self.assertEqual(len({spec[0] for spec in images.SPECS}), 12)
        for _, source, box, _ in images.SPECS:
            if source != "api_terminal":
                self.assertLessEqual(box[2] - box[0], 700)
                self.assertLessEqual(box[3] - box[1], 650 if "mobile" in source else 430)


if __name__ == "__main__":
    unittest.main()
