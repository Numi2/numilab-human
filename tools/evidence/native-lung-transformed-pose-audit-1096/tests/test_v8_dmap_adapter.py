import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/"source"
E = Path("/Users/n/numi-human-resting-evidence-20261005")
V8 = E / "native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8"
NHA = V8 / "final/resting-thorax.nhanatomy"
NHA_SHA = "1b62f569f218cf8ba45f26c393bfaa4c8ed2ff499ab500c002dbbb006747b8bc"
COMP = V8 / "composition-report.json"

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class V8DMapAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load(ROOT / "audit_lung_cycle_1096.py", "runner1096_test")
        cls.base = cls.runner.load_module(cls.runner.require_hash(cls.runner.BASE, cls.runner.BASE_SHA), "base1096_test")
        parser = cls.base.load(cls.base.PARSER, "parser1096_test")
        cls.rows = parser.parse_payload(NHA)[1]
        cls.adapter = load(ROOT / "v8_dmap_adapter.py", "v8adapter1096_test")

    def test_real_v8_map_is_current_and_reciprocal(self):
        doc = self.adapter.load_v8_current_dmap(
            base=self.base, composition_report_path=COMP,
            final_nha_path=NHA, final_nha_sha=NHA_SHA, final_rows=self.rows)
        self.assertEqual(doc["map_count"], 47343)
        self.assertEqual(doc["counts_by_lobe"], {305: 19743, 306: 21600, 307: 5514, 308: 486})
        self.assertEqual(doc["zero_lobes"], [309])
        self.assertEqual(set(doc["pairs"]), {305, 306, 307, 308, 309})
        for sid, pair in doc["pairs"].items():
            self.assertEqual(pair["map_count"], {305: 19743, 306: 21600, 307: 5514, 308: 486, 309: 0}[sid])
            self.assertEqual(len(pair["d2l"]), len(pair["l2d"]))
        self.assertLess(abs(doc["registered_area_m2"] - doc["map_area_sum_m2"]), 1e-12)

    def test_wrong_payload_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, "caller-selected current NHA"):
            self.adapter.load_v8_current_dmap(
                base=self.base, composition_report_path=COMP,
                final_nha_path=NHA, final_nha_sha="0" * 64, final_rows=self.rows)

if __name__ == "__main__":
    unittest.main()
