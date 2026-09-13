from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from numilab_human.cardiac_body_links import SCHEMA, compile_body_links
from numilab_human.model import ImportError as HumanImportError
from numilab_human.physiology import canonical


ROOT = Path(__file__).resolve().parents[1]


class CardiacBodyLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = compile_body_links()

    def test_four_cavities_bind_to_torso_without_physical_promotion(self) -> None:
        result = self.result
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(len(result["bindings"]), 4)
        self.assertEqual(
            [(row["member_id"], row["source_name"], row["source_body_id"], row["core_body_index"])
             for row in result["bindings"]],
            [("FJ2424", "cavity of right atrium", 9, 20),
             ("FJ2423", "cavity of right ventricle", 9, 20),
             ("FJ2425", "cavity of left atrium", 9, 20),
             ("FJ2422", "cavity of left ventricle", 9, 20)],
        )
        self.assertTrue(result["qualification"]["body_link_registration"])
        self.assertFalse(result["qualification"]["physical_volume_authority_assigned"])
        self.assertFalse(result["qualification"]["blood_mass_assigned"])
        self.assertFalse(result["qualification"]["cavity_domains_disjoint"])
        for row in result["bindings"]:
            self.assertTrue(row["body_link_registration"])
            self.assertIsNone(row["physical_volume_owner"])
            self.assertIsNone(row["mechanical_mass_owner"])
            self.assertIsNone(row["density_kg_per_m3"])

    def test_cavity_map_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cavity-map.json"
            mapping = json.loads((ROOT / "config/bodyparts3d-myosim-cardiac-cavity-map.v1.json").read_text())
            mapping["entries"][0]["myosim_body"] = "Abdomen"
            path.write_bytes(canonical(mapping) + b"\n")
            with self.assertRaisesRegex(HumanImportError, "body link changed"):
                compile_body_links(cavity_map=path)

    def test_cavity_bridge_hash_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bridge.json"
            bridge = json.loads((ROOT / "Docs/media/organ-blood-cavity-bridge-20260913/bridge.json").read_text())
            bridge["bindings"][0]["source_member_sha256"] = "x" * 64
            path.write_bytes(canonical(bridge) + b"\n")
            with self.assertRaisesRegex(HumanImportError, "bridge source member hash"):
                compile_body_links(bridge=path)


if __name__ == "__main__":
    unittest.main()
