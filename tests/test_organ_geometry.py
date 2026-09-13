from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from numilab_human.model import ImportError as HumanImportError
from numilab_human.organ_geometry import (
    ARCHIVE, ROOT, SCHEMA, _immutable_write, inventory,
)
from numilab_human.physiology import canonical, read_json


class OrganGeometryInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = inventory()

    def test_all_template_regions_and_members_are_source_bound(self) -> None:
        result = self.result
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["counts"], {
            "region_count": 18,
            "member_count": 378,
            "declared_membership_count": 386,
            "closed_quotient_member_count": 371,
            "open_or_defective_quotient_member_count": 7,
            "region_with_open_or_defective_member_count": 3,
        })
        self.assertTrue(result["qualification"]["source_membership_verified"])
        self.assertTrue(result["qualification"]["exact_coordinate_seam_quotient_performed"])
        self.assertFalse(result["qualification"]["watertight_union_performed"])
        self.assertFalse(result["qualification"]["blood_mass_assigned"])
        self.assertEqual(sum(row["member_count"] for row in result["regions"]), 386)

    def test_defective_source_members_are_retained_and_named(self) -> None:
        result = self.result
        defective = {
            member["member_id"]
            for region in result["regions"]
            for member in region["surface_members"]
            if not member["topology"]["closed_oriented_manifold_candidate"]
        }
        self.assertEqual(defective, {"FJ2434", "FJ2928", "FJ2404", "FJ2405", "FJ2409", "FJ2820", "FJ2821"})
        self.assertEqual(
            {region["id"] for region in result["regions"]
             if region["open_or_defective_quotient_member_count"]},
            {"right_ventricle", "left_lung", "liver"},
        )
        self.assertTrue(all(member["repair_applied"] is False
                            for region in result["regions"] for member in region["surface_members"]))

    def test_manifest_is_deterministic_and_immutable(self) -> None:
        first = canonical(self.result)
        self.assertEqual(first, canonical(inventory()))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "inventory.json"
            self.assertEqual(_immutable_write(output, self.result), _immutable_write(output, self.result))
            output.write_bytes(b'{"forged":true}\n')
            with self.assertRaisesRegex(HumanImportError, "immutable"):
                _immutable_write(output, self.result)

    def test_archive_lock_rebinding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock = read_json(ROOT / "sources.lock.json")
            lock["sources"]["bodyparts3d_4"]["files"][ARCHIVE]["sha256"] = "0" * 64
            lock_path = Path(temporary) / "sources.lock.json"
            lock_path.write_bytes(canonical(lock))
            with self.assertRaisesRegex(HumanImportError, "hash mismatch"):
                inventory(source_lock=lock_path)

    def test_template_membership_rebinding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            template = read_json(ROOT / "config/physiology-organ-network-template.v1.json")
            template["regions"][0] = copy.deepcopy(template["regions"][0])
            template["regions"][0]["member_ids"] = ["FJ9999"]
            template_path = Path(temporary) / "template.json"
            template_path.write_bytes(canonical(template))
            with self.assertRaisesRegex(HumanImportError, "complete source membership"):
                inventory(template=template_path)


if __name__ == "__main__":
    unittest.main()
