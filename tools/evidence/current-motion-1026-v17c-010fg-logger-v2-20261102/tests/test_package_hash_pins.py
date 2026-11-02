import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublishedSourcePinTests(unittest.TestCase):
    def test_all_published_sources_match_pin_manifest(self):
        pins = json.loads((ROOT / "source-pins.json").read_text(encoding="utf-8"))
        for entry in pins["published_artifacts"]:
            path = ROOT / entry["path"]
            self.assertTrue(path.is_file(), entry["path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, entry["sha256"], entry["path"])


if __name__ == "__main__":
    unittest.main()
