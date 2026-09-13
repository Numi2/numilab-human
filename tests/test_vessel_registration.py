from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from numilab_human.model import ImportError as HumanImportError
from numilab_human.physiology import canonical
from numilab_human.vessel_registration import (
    SCHEMA, _immutable_write, compile_registration,
)


class VesselRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = compile_registration()

    def test_six_vessels_are_source_bound_to_world_without_mechanical_promotion(self) -> None:
        result = self.result
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(len(result["bindings"]), 6)
        self.assertEqual(
            [row["member_id"] for row in result["bindings"]],
            ["FJ1932", "FJ3411", "FJ3413", "FJ3427", "FJ3441", "FJ3645"],
        )
        self.assertTrue(result["qualification"]["source_to_world_frame_registered"])
        self.assertFalse(result["qualification"]["body_link_registration"])
        self.assertFalse(result["qualification"]["tubular_vessel_field"])
        for row in result["bindings"]:
            self.assertTrue(row["world_frame_registration"])
            self.assertFalse(row["body_link_registration"])
            self.assertFalse(row["tubular_field_registered"])
            self.assertIsNone(row["cross_section_area_m2"])
            self.assertIsNone(row["material_density_kg_per_m3"])
            self.assertIsNone(row["mechanical_mass_owner"])
            self.assertGreater(row["registered_world_surface_integral_volume_m3"], 0.0)

    def test_deterministic_and_immutable(self) -> None:
        self.assertEqual(canonical(self.result), canonical(compile_registration()))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "registration.json"
            self.assertEqual(_immutable_write(output, self.result), _immutable_write(output, self.result))
            output.write_bytes(b'{"forged":true}\n')
            with self.assertRaisesRegex(HumanImportError, "immutable"):
                _immutable_write(output, self.result)

    def test_registration_archive_rebinding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registration.json"
            tampered = json.loads(Path(self.result["source"]["registration"]).read_text())
            for row in tampered["source"]["bodyparts"]["archives"]:
                if row.get("hierarchy") == "part_of":
                    row["sha256"] = "0" * 64
            path.write_bytes(canonical(tampered) + b"\n")
            with self.assertRaisesRegex(HumanImportError, "archive provenance"):
                compile_registration(registration=path)

    def test_registration_transform_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registration.json"
            tampered = json.loads(Path(self.result["source"]["registration"]).read_text())
            tampered["coordinate_system"]["global_source_mm_to_myosim_world_m"][0][1] = 1.0e-6
            path.write_bytes(canonical(tampered) + b"\n")
            with self.assertRaisesRegex(HumanImportError, "signed axis permutation"):
                compile_registration(registration=path)


if __name__ == "__main__":
    unittest.main()
