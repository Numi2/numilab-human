from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from numilab_human.model import ImportError as HumanImportError
from numilab_human.organ_blood_bridge import (
    SCHEMA, _immutable_write, compile_bridge,
)
from numilab_human.physiology import canonical


class OrganBloodBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = compile_bridge()

    def test_exact_four_source_bound_chambers_and_unresolved_owner(self) -> None:
        result = self.result
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(len(result["bindings"]), 4)
        self.assertEqual([row["source_index"] for row in result["bindings"]], [15, 16, 19, 20])
        self.assertEqual([row["member_id"] for row in result["bindings"]],
                         ["FJ2424", "FJ2423", "FJ2425", "FJ2422"])
        for row in result["bindings"]:
            self.assertEqual(row["source_moment_status"], "computed_single_closed_component")
            self.assertGreater(row["source_surface_integral_volume_m3"], 0)
            self.assertGreater(row["initial_hydraulic_to_source_integral_volume_ratio"], 0)
            self.assertIsNone(row["physical_volume_owner"])
            self.assertIsNone(row["mechanical_mass_owner"])
            self.assertIsNone(row["density_kg_per_m3"])
            self.assertFalse(row["world_or_body_frame_registration"])
            self.assertFalse(row["pressure_gradient_momentum_transfer"])

    def test_identity_and_cavity_boundary_are_explicit(self) -> None:
        result = self.result
        self.assertTrue(result["qualification"]["exact_source_member_hash_binding"])
        self.assertTrue(result["qualification"]["source_frame_moments_reused"])
        self.assertFalse(result["qualification"]["physical_volume_authority_assigned"])
        self.assertFalse(result["qualification"]["blood_mass_assigned"])
        self.assertFalse(result["qualification"]["world_or_body_frame_registered"])
        self.assertFalse(result["qualification"]["material_density_calibrated"])
        self.assertFalse(result["qualification"]["physiological_calibration"])
        self.assertFalse(result["qualification"]["standing_walking"])
        self.assertFalse(result["cvsim_cavity_reference"]["all_cavity_domains_disjoint"])
        self.assertGreater(result["cvsim_cavity_reference"]["intersecting_triangle_pairs"], 0)

    def test_deterministic_and_immutable(self) -> None:
        self.assertEqual(canonical(self.result), canonical(compile_bridge()))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bridge.json"
            self.assertEqual(_immutable_write(output, self.result), _immutable_write(output, self.result))
            output.write_bytes(b'{"forged":true}\n')
            with self.assertRaisesRegex(HumanImportError, "immutable"):
                _immutable_write(output, self.result)

    def test_stale_or_tampered_moments_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "moments.json"
            tampered = copy.deepcopy(json.loads(
                (Path("Docs/media/organ-geometry-moments-20260913/moments.json")).read_text()
            ))
            tampered["members"][0]["source_sha256"] = "0" * 64
            path.write_bytes(canonical(tampered) + b"\n")
            with self.assertRaisesRegex(HumanImportError, "does not match"):
                compile_bridge(moments=path)


if __name__ == "__main__":
    unittest.main()
