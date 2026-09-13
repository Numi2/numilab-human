from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from numilab_human.cardiac_loading_contract import (
    CONFIG, SCHEMA, _immutable_write, compile_contract,
)
from numilab_human.model import ImportError as HumanImportError
from numilab_human.physiology import canonical


class CardiacLoadingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = compile_contract()

    def test_source_material_activation_and_loading_are_retained(self) -> None:
        result = self.result
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["source"]["config_sha256"],
                         "23c931fef53edced85e0e0a36c73d8490dddb87db6bd988482a2cdfc5a1442cc")
        self.assertEqual(result["source_activation"]["parameters"]["peak_isometric_tension"],
                         {"value": 120, "unit": "kPa"})
        self.assertEqual(result["source_loading"]["initial_pressures"]["lv_endocardial"],
                         {"value": 1.6, "unit": "kPa"})
        self.assertEqual(result["source_loading"]["valve_resistance"]["aortic_forward"]["value"], 0)
        self.assertEqual(result["closure_labels"], list(range(11, 25)))
        self.assertTrue(result["qualification"]["source_parameters_hash_bound"])
        self.assertTrue(result["qualification"]["passive_material_classes_retained"])

    def test_unresolved_support_loading_activation_and_material_gates_stay_open(self) -> None:
        result = self.result
        self.assertEqual([gate["id"] for gate in result["unresolved_gates"]], [
            "stress_free_reference", "inertial_density", "epicardial_robin_coefficients",
            "venous_anchor_interpretation", "source_reference_trajectory", "zero_forward_valve_admission",
        ])
        self.assertTrue(all(gate["admitted"] is False for gate in result["unresolved_gates"]))
        for key in ("quantitative_robin_support_admitted", "venous_anchor_support_admitted",
                    "stress_free_reference_supplied", "source_activation_time_field_supplied",
                    "closure_materials_resolved", "inertial_density_assigned",
                    "native_anatomical_wall_admitted", "subject_specific_calibration", "physical_steps"):
            self.assertFalse(result["qualification"][key])
        self.assertFalse(result["source_qualification"]["subject_specific_material_calibration"])
        self.assertFalse(result["source_qualification"]["native_anatomical_coupling_qualified"])

    def test_deterministic_and_immutable(self) -> None:
        self.assertEqual(canonical(self.result), canonical(compile_contract()))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            self.assertEqual(_immutable_write(path, self.result), _immutable_write(path, self.result))
            path.write_bytes(b'{"forged":true}\n')
            with self.assertRaisesRegex(HumanImportError, "immutable"):
                _immutable_write(path, self.result)

    def test_configuration_hash_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            config = json.loads(CONFIG.read_text())
            config["id"] = "forged"
            path.write_bytes(canonical(config))
            with self.assertRaisesRegex(HumanImportError, "hash mismatch"):
                compile_contract(config=path)


if __name__ == "__main__":
    unittest.main()
