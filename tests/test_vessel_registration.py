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
from numilab_human.vessel_body_links import (
    SCHEMA as BODY_LINK_SCHEMA, compile_body_links,
)


ROOT = Path(__file__).resolve().parents[1]


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

    def test_exact_myosim_body_links_are_hash_bound_without_mechanical_promotion(self) -> None:
        manifest = ROOT / "Docs/media/organ-vessel-body-link-20260913/authoritative/myosim-fullbody-reference.manifest.json"
        receipt = ROOT / "Docs/media/organ-vessel-registration-20260913/registration.json"
        result = compile_body_links(registration=receipt, human_manifest=manifest)
        self.assertEqual(result["schema"], BODY_LINK_SCHEMA)
        self.assertEqual(
            [(row["member_id"], row["myosim_body"], row["source_body_id"], row["core_body_index"])
             for row in result["bindings"]],
            [("FJ1932", "Abdomen", 4, 7), ("FJ3411", "torso", 9, 20),
             ("FJ3413", "torso", 9, 20), ("FJ3427", "torso", 9, 20),
             ("FJ3441", "Abdomen", 4, 7), ("FJ3645", "torso", 9, 20)],
        )
        self.assertTrue(result["qualification"]["body_link_registration"])
        self.assertFalse(result["qualification"]["tubular_vessel_field"])
        self.assertFalse(result["qualification"]["blood_mass_owner"])
        self.assertFalse(result["qualification"]["subject_calibration"])
        for row in result["bindings"]:
            self.assertTrue(row["body_link_registration"])
            self.assertIsNone(row["mechanical_mass_owner"])
            self.assertIsNone(row["material_density_kg_per_m3"])

    def test_body_link_manifest_tampering_is_rejected(self) -> None:
        manifest = ROOT / "Docs/media/organ-vessel-body-link-20260913/authoritative/myosim-fullbody-reference.manifest.json"
        receipt = ROOT / "Docs/media/organ-vessel-registration-20260913/registration.json"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / manifest.name
            tampered = json.loads(manifest.read_text())
            tampered["payloads"]["rigid"]["sha256"] = "0" * 64
            path.write_bytes(canonical(tampered) + b"\n")
            rigid = path.parent / tampered["payloads"]["rigid"]["file"]
            rigid.write_bytes((manifest.parent / "myosim-fullbody-core-reference.nhrigid").read_bytes())
            with self.assertRaisesRegex(HumanImportError, "manifest"):
                compile_body_links(registration=receipt, human_manifest=path)

    def test_body_link_source_mapping_tampering_is_rejected(self) -> None:
        manifest = ROOT / "Docs/media/organ-vessel-body-link-20260913/authoritative/myosim-fullbody-reference.manifest.json"
        receipt = ROOT / "Docs/media/organ-vessel-registration-20260913/registration.json"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / receipt.name
            tampered = json.loads(receipt.read_text())
            next(row for row in tampered["bindings"] if row["member_id"] == "FJ1932")["myosim_body"] = "torso"
            path.write_bytes(canonical(tampered) + b"\n")
            with self.assertRaisesRegex(HumanImportError, "pinned anatomy map"):
                compile_body_links(registration=path, human_manifest=manifest)


if __name__ == "__main__":
    unittest.main()
