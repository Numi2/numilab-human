from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from numilab_human.model import ImportError as HumanImportError
from numilab_human.organ_geometry_component_moments import (
    ROOT,
    SCHEMA,
    _immutable_write,
    compile_component_moments,
)
from numilab_human.physiology import canonical, read_json


class OrganGeometryComponentMomentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = compile_component_moments()

    def test_status_counts_and_source_boundary(self) -> None:
        result = self.result
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["counts"]["region_count"], 18)
        self.assertEqual(result["counts"]["member_count"], 378)
        self.assertEqual(result["counts"]["moment_computed_member_count"], 364)
        self.assertEqual(result["status_counts"], {
            "computed_disjoint_closed_component_sum": 7,
            "computed_single_closed_component": 357,
            "not_disjoint_component_bounds": 7,
            "source_topology_defective": 7,
        })
        self.assertTrue(result["qualification"]["disjoint_closed_component_sums_computed"])
        self.assertFalse(result["qualification"]["physical_volume_authority_assigned"])
        self.assertFalse(result["qualification"]["blood_mass_assigned"])

    def test_only_disjoint_multicomponent_members_are_integrated(self) -> None:
        result = self.result
        aggregate_ids = {
            row["member_id"] for row in result["members"]
            if row["moment_status"] == "computed_disjoint_closed_component_sum"
        }
        self.assertEqual(aggregate_ids, {
            "FJ1893", "FJ3090", "FJ3093", "FJ3110", "FJ3113", "FJ3115", "FJ3116",
        })
        for row in result["members"]:
            if row["moment_status"] == "computed_disjoint_closed_component_sum":
                self.assertTrue(row["component_aabb_disjoint_checked"])
                self.assertTrue(row["component_aabb_disjoint"])
                self.assertEqual(len(row["source_component_moments"]), row["face_component_count"])
                self.assertGreater(row["source_surface_moments"]["absolute_signed_volume_m3"], 0.0)
                self.assertIsNone(row["physical_volume_m3"])
                self.assertIsNone(row["mechanical_mass_kg"])
            elif row["moment_status"] == "not_disjoint_component_bounds":
                self.assertFalse(row["component_aabb_disjoint"])
                self.assertIsNone(row["source_surface_moments"])

    def test_receipt_is_deterministic_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "component-moments.json"
            digest = _immutable_write(output, self.result)
            self.assertEqual(digest, _immutable_write(output, self.result))
            output.write_bytes(b'{"forged":true}\n')
            with self.assertRaisesRegex(HumanImportError, "immutable"):
                _immutable_write(output, self.result)
        receipt = read_json(ROOT / "Docs/media/organ-geometry-component-moments-20260915/receipt-v1.json")
        self.assertEqual(canonical(receipt), canonical(self.result))


if __name__ == "__main__":
    unittest.main()
