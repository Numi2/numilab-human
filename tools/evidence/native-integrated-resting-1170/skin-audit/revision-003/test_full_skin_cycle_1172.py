import importlib.util
import unittest
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "audit_full_skin_cycle_1172.py"
spec = importlib.util.spec_from_file_location("skin_audit_1172_tests", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class RegisteredFullSkinAuditTests(unittest.TestCase):
    def test_terminal_schedule_requires_eight_unique_steps_and_exact_final_N(self):
        steps = [47519, 49151, 51903, 54047, 55647, 152191, 154143, 155000]
        self.assertEqual(audit.validate_steps(steps, 155000), steps)
        for bad, terminal in [([1, 2, 3, 4, 5, 6, 7], 155000),
                              ([1, 2, 3, 4, 5, 6, 7, 7], 155000),
                              ([1, 2, 3, 4, 5, 6, 155000, 154999], 155000),
                              ([1, 2, 3, 4, 5, 6, 7, 10000], 10000)]:
            with self.assertRaises(ValueError): audit.validate_steps(bad, terminal)

    def test_exact_927_skin_asset_and_manifest_chain(self):
        proof = audit.validate_skin_asset()
        self.assertEqual(proof["payload_sha256"], audit.SKIN_SHA)
        self.assertEqual(proof["registration_sha256"], audit.SKIN_REGISTRATION_SHA)
        self.assertEqual(proof["historical_composition_module_sha256"], audit.SKIN_MODULE_SHA)
        self.assertTrue(proof["historical_composition_module_git_commit"])

    def test_all_859_targets_and_all_17_eye_monitors_are_present(self):
        keys, rows = audit.inventory()
        self.assertEqual(len(keys), 859)
        self.assertEqual(len(rows), 859)
        self.assertTrue(audit.OCULAR.issubset(set(keys)))

    def test_registered_trial_adapter_is_hash_pinned_and_no_runmetadata_dependency(self):
        self.assertEqual(audit.sha(audit.REGISTERED_ADAPTER), audit.REGISTERED_ADAPTER_SHA)
        src = SCRIPT.read_text()
        self.assertIn("load_registered_context", src)
        self.assertIn("scene_run_metadata_required", src)
        self.assertNotIn('scene / "run-metadata.json"', src)
        adapter_src = audit.REGISTERED_ADAPTER.read_text()
        self.assertIn('P18 = EVIDENCE / "final-integrated-study-readiness-1170/package-v018"', adapter_src)
        for key in ("p18_analyzer", "p18_prepare", "p18_helper_modules"):
            self.assertIn(key, adapter_src)

    def test_pack_inventory_allows_only_required_bones_skin_and_all_targets(self):
        keys, _ = audit.inventory()
        expected = audit.expected_pack_surface_keys(keys)
        support = {(51004, stable_id) for stable_id in range(1, 186)}
        self.assertEqual(len(keys), 859)
        self.assertEqual(len(support), 185)
        audit.validate_pack_surface_keys({key: None for key in expected}, keys)
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            audit.validate_pack_surface_keys({key: None for key in expected if key != (51004, 1)}, keys)
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            audit.validate_pack_surface_keys({key: None for key in expected} | {(99999, 1): None}, keys)
        target_only = next(key for key in keys if key[0] != 51004)
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            audit.validate_pack_surface_keys({key: None for key in expected if key != target_only}, keys)

    def test_predicates_use_pinned_integration_package_even_if_p18_package_is_loaded(self):
        prior = sys.modules.get("numilab_human")
        foreign = types.ModuleType("numilab_human")
        foreign.__path__ = ["/Users/n/numi-human-lung-triangulation-candidate-003/src/numilab_human"]
        sys.modules["numilab_human"] = foreign
        try:
            ci, clearance, _, _ = audit.load_predicates()
            self.assertEqual(Path(ci.__file__).resolve(), audit.PRED.resolve())
            self.assertEqual(Path(clearance.__file__).resolve(), audit.CLEARANCE.resolve())
            self.assertTrue(ci.__name__.startswith(audit.PREDICATE_PACKAGE_NAME + "."))
        finally:
            if prior is None:
                sys.modules.pop("numilab_human", None)
            else:
                sys.modules["numilab_human"] = prior
            for name in tuple(sys.modules):
                if name == audit.PREDICATE_PACKAGE_NAME or name.startswith(audit.PREDICATE_PACKAGE_NAME + "."):
                    del sys.modules[name]

if __name__ == "__main__": unittest.main()
