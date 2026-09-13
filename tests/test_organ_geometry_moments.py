from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from numilab_human.model import ImportError as HumanImportError
from numilab_human.organ_geometry_moments import (
    SCHEMA,
    _immutable_write,
    compile_moments,
)
from numilab_human.physiology import canonical


class OrganGeometryMomentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = compile_moments()

    def test_status_counts_and_source_boundary(self) -> None:
        result = self.result
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["counts"]["region_count"], 18)
        self.assertEqual(result["counts"]["member_count"], 378)
        self.assertEqual(result["counts"]["closed_quotient_member_count"], 371)
        self.assertEqual(result["counts"]["moment_computed_member_count"], 357)
        self.assertEqual(result["counts"]["not_single_closed_component_member_count"], 14)
        self.assertEqual(result["counts"]["source_topology_defective_member_count"], 7)
        self.assertEqual(result["status_counts"], {
            "computed_single_closed_component": 357,
            "not_single_closed_component": 14,
            "source_topology_defective": 7,
        })
        self.assertTrue(result["qualification"]["single_closed_surface_integrals_computed"])
        self.assertFalse(result["qualification"]["physical_volume_authority_assigned"])
        self.assertFalse(result["qualification"]["blood_mass_assigned"])

    def test_computed_moments_are_finite_and_unpromoted(self) -> None:
        computed = [row for row in self.result["members"]
                    if row["moment_status"] == "computed_single_closed_component"]
        self.assertEqual(len(computed), 357)
        for row in computed:
            moments = row["source_surface_moments"]
            self.assertGreater(moments["absolute_signed_volume_m3"], 0.0)
            self.assertTrue(all(value == value and abs(value) != float("inf")
                                for value in moments["centroid_source_frame_m"]))
            self.assertIsNone(moments["physical_volume_m3"])
            self.assertIsNone(moments["mechanical_mass_kg"])
            self.assertIsNone(row["physical_volume_m3"])
            self.assertIsNone(row["mechanical_mass_kg"])

    def test_defects_and_multi_component_candidates_remain_explicit(self) -> None:
        defective = {row["member_id"] for row in self.result["members"]
                     if row["moment_status"] == "source_topology_defective"}
        self.assertEqual(defective, {"FJ2434", "FJ2928", "FJ2404", "FJ2405", "FJ2409", "FJ2820", "FJ2821"})
        multi = [row for row in self.result["members"]
                 if row["moment_status"] == "not_single_closed_component"]
        self.assertEqual(len(multi), 14)
        self.assertTrue(all(row["source_surface_moments"] is None for row in multi))

    def test_result_is_deterministic_and_immutable(self) -> None:
        self.assertEqual(canonical(self.result), canonical(compile_moments()))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "moments.json"
            self.assertEqual(_immutable_write(output, self.result), _immutable_write(output, self.result))
            output.write_bytes(b'{"forged":true}\n')
            with self.assertRaisesRegex(HumanImportError, "immutable"):
                _immutable_write(output, self.result)


if __name__ == "__main__":
    unittest.main()
