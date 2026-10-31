import importlib.util
import unittest
from pathlib import Path
import numpy as np

MODULE_PATH = Path(__file__).with_name("prepare_registered_deep_authoring_view_593_v6.py")
spec = importlib.util.spec_from_file_location("authoring_adapter_v6", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ProjectionControls(unittest.TestCase):
    def fixture(self):
        schema = {
            "5": {"role": "D", "layer": 2},
            "7": {"role": "BONE", "face_count": 2},
            "90": {"role": "DERIVED_PATCH", "layer": 3},
            "99": {"role": "D", "derived_operation": "mixed_parent_control", "derived_parent_source_ids": [5, 7]},
        }
        candidate = {
            "face_source_original_id": np.asarray([5, 7, 99, 90], dtype=np.int64),
            "face_source_layer": np.asarray([2, 3, 3, 3], dtype=np.int64),
            "quality_star_patch_union_face_source_original_id": np.asarray([99, 99, 7, 90], dtype=np.int64),
            "quality_star_patch_union_face_source_layer": np.asarray([2, 3, 3, 3], dtype=np.int64),
        }
        return candidate, schema

    def test_role_D_mixed_source_layers_project_separately(self):
        candidate, schema = self.fixture()
        out, report = module.project_layers(candidate, schema)
        np.testing.assert_array_equal(out["face_source_layer"], [2, 3, 2, 3])
        np.testing.assert_array_equal(out["quality_star_patch_union_face_source_layer"], [2, 2, 3, 3])
        self.assertEqual(report["source_registry_projection_ids"], [99])
        self.assertEqual(report["per_source_projection"][0]["observed_source_layer_codes"], [2, 3])
        self.assertEqual(report["per_source_projection"][0]["authoring_layer_code"], 2)
        self.assertEqual(out["face_source_layer"][1], 3)  # registered BONE provenance is untouched
        self.assertEqual(out["face_source_layer"][3], 3)  # DERIVED_PATCH is untouched
        np.testing.assert_array_equal(candidate["face_source_layer"], [2, 3, 3, 3])


    def test_inactive_registry_derivative_is_reported_not_projected(self):
        candidate, schema = self.fixture()
        schema["101"] = {"role": "D", "derived_operation": "inactive_fixture"}
        out, report = module.project_layers(candidate, schema)
        self.assertEqual(report["inactive_registry_derived_D_ids_without_candidate_tuple_rows"], [101])
        np.testing.assert_array_equal(out["face_source_layer"], [2, 3, 2, 3])


    def test_inactive_registry_derivative_is_reported_not_projected(self):
        candidate, schema = self.fixture()
        schema["101"] = {"role": "D", "derived_operation": "inactive_fixture"}
        out, report = module.project_layers(candidate, schema)
        self.assertEqual(report["inactive_registry_derived_D_ids_without_candidate_tuple_rows"], [101])
        np.testing.assert_array_equal(out["face_source_layer"], [2, 3, 2, 3])

    def test_unsupported_source_tuple_fails_closed(self):
        candidate, schema = self.fixture()
        candidate["face_source_layer"][2] = 4
        with self.assertRaises(module.AdapterError):
            module.project_layers(candidate, schema)

    def test_BONE_parent_must_remain_source_layer_3(self):
        candidate, schema = self.fixture()
        candidate["face_source_layer"][1] = 2
        with self.assertRaises(module.AdapterError):
            module.project_layers(candidate, schema)

if __name__ == "__main__":
    unittest.main()
