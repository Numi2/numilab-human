#!/usr/bin/env python3
import hashlib
import importlib.util
import unittest
from pathlib import Path

E = Path("/Users/n/numi-human-resting-evidence-20261005")
PKG = E / "native-lung-late-pose-audit-runner-1171"
NHA = E / "native-lung-free-apex-composition-1159-eightops-attempt1/final/resting-thorax.nhanatomy"
COMPOSITION = E / "native-lung-free-apex-composition-1159-eightops-attempt1/composition-report.json"
LL_REPORT = E / "native-lung-free-apex-composition-1159-eightops-attempt1/current-reciprocal-map-report-v2.json"
CONFIG = E / "native-lung-free-apex-composition-1159-eightops-attempt1/final/resting-reference-respiration.json"
CIRCULATION = E / "reference-circulation-001/resting_reference_lv15.native.v3.json"
EXPECTED_NHA = "c10dce4609be99fdc569801c2705e23c46120b610c8aa86c631165d38baf4713"
EXPECTED_COMPOSITION = "ce5948d2dfe6388fc64f656fa34abfd7f409e471ee272a4a8231eba38281e108"
EXPECTED_LL_REPORT = "fd054aea9f3812ca50604854348bbd7c75c6acfeb3c4678ba418b50eaf6d331f"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class LatePose1159Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load(PKG / "audit_lung_cycle_1159.py", "audit_lung_cycle_1159_test")
        cls.base = load(cls.runner.BASE, "native_lung_exact_base_1159_test")
        cls.parser = load(cls.base.PARSER, "native_lung_parser_1159_test")
        cls.rows = cls.parser.parse_payload(NHA)[1]

    def test_local_successor_and_registered_adapter_pins(self):
        self.assertEqual(self.runner.HERE, PKG)
        self.assertEqual(sha(PKG / "registered_receipt_adapter_1171.py"),
                         self.runner.REGISTERED_ADAPTER_SHA)
        self.assertEqual(sha(PKG / "successor_dmap_adapter_1159.py"),
                         self.runner.V8_DMAP_ADAPTER_SHA)
        self.assertEqual(sha(PKG / "v8_dmap_adapter_1096.py"),
                         "9a55338d874ecc4695e1545829c21266b9b5734d8ca8d64acab817906091c7bd")

    def test_capture_contract_accepts_only_p17_eight_steps_with_true_terminal(self):
        control = [47519, 49151, 51903, 54047, 55647, 152191, 154143, 155000]
        treatment = [47519, 49151, 51903, 54047, 55647, 152447, 154367, 155000]
        self.assertEqual(self.runner.validate_terminal_capture(control, 155000)["accepted_steps"], control)
        self.assertEqual(self.runner.validate_terminal_capture(treatment, 155000)["accepted_steps"], treatment)
        for bad, roots in ((control[:-1], 155000), (control[:-1] + [154999], 155000),
                           (control[:-1] + [154143], 155000), (control[:-1] + [155000], 154999)):
            with self.assertRaises(ValueError):
                self.runner.validate_terminal_capture(bad, roots)

    def test_current_1159_d_and_lobe_maps_rebind_exactly(self):
        self.assertEqual(sha(NHA), EXPECTED_NHA)
        self.assertEqual(sha(COMPOSITION), EXPECTED_COMPOSITION)
        self.assertEqual(sha(LL_REPORT), EXPECTED_LL_REPORT)
        adapter = load(PKG / "successor_dmap_adapter_1159.py", "current_1159_d_map_adapter_test")
        dmap = adapter.load_v8_current_dmap(base=self.base, composition_report_path=COMPOSITION,
            final_nha_path=NHA, final_nha_sha=EXPECTED_NHA, final_rows=self.rows)
        self.assertEqual(dmap["map_count"], 47343)
        self.assertEqual(dmap["counts_by_lobe"], {305:19743, 306:21600, 307:5514, 308:486, 309:0})
        self.assertEqual(dmap["zero_lobes"], [309])
        self.assertTrue(dmap["registered_face_validation"]["exact_xyz_triangle_match"])
        self.assertTrue(dmap["registered_face_validation"]["exact_face_index_triple_match"])
        lineage = load(PKG / "lobe_lineage_v2.py", "current_1159_lobe_lineage_test")
        ll = lineage.load_v2_bridge(base=self.base, report_path=LL_REPORT, final_nha_path=NHA,
                                    final_nha_sha=EXPECTED_NHA, final_rows=self.rows)
        self.assertEqual(len(ll["pairs"]), 10)
        self.assertEqual(sum(x["map_count"] for x in ll["pairs"].values()), 29226)

    def test_v8_adapter_rejects_wrong_selected_nha_hash(self):
        adapter = load(PKG / "v8_dmap_adapter_1096.py", "strict_1171_v8_parent_adapter_test")
        parent_comp = E / "native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8/composition-report.json"
        parent_nha = parent_comp.parent / "final/resting-thorax.nhanatomy"
        parent_rows = self.parser.parse_payload(parent_nha)[1]
        with self.assertRaisesRegex(ValueError, "not bound to caller-selected current NHA"):
            adapter.load_v8_current_dmap(
                base=self.base, composition_report_path=parent_comp,
                final_nha_path=parent_nha, final_nha_sha="0" * 64, final_rows=parent_rows,
            )

    def test_effective_area_and_per_lobe_config_binding_is_current(self):
        invocation = {"argv":["native", "--resting-scene", str(CIRCULATION), str(CONFIG)],
                      "asset_sha256":{str(CONFIG):sha(CONFIG), str(CIRCULATION):sha(CIRCULATION)}}
        area = self.runner.respiration_area_binding(self.base, invocation, NHA, EXPECTED_NHA, self.rows)
        self.assertEqual(area["status"], "PASS_exact_geometry_and_config_binding")
        self.assertEqual(area["mismatch_fields"], [])

if __name__ == "__main__":
    unittest.main()
