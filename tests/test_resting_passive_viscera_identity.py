import unittest
from copy import deepcopy
import hashlib

from numilab_human.resting_passive_viscera import (
    COLON_COMPONENTS,
    SMALL_INTESTINE_MEMBERS,
    anatomical_component_identity,
    refresh_passive_component_identities,
    vascular_bindings,
)


def _refresh_fixture():
    small_components = {
        "FJ2573": ("FMA7206", "duodenum"),
        "FJ2574": ("FMA14964", "proximal part of ileum"),
        "FJ2599": ("FMA11338", "ileocecal junction"),
        "FJ2627": ("FMA16981", "proximal part of jejunum"),
    }
    tables = {"is_a": {}, "part_of": {}}
    for member in SMALL_INTESTINE_MEMBERS:
        component = small_components.get(member, ("FMA14965", "middle part of ileum"))
        tables["is_a"].setdefault(component, set()).add(member)
    for member, component in COLON_COMPONENTS.items():
        tables["part_of"].setdefault(component, set()).add(member)
    atlas = {
        "source": {
            "id": "bodyparts3d_4", "version": "4.0", "license": "CC-BY-4.0",
            "attribution": "BodyParts3D", "url": "https://example.invalid/bodyparts3d",
            "tables": [
                {"file": "isa_element_parts.txt", "sha256": "a" * 64},
                {"file": "partof_element_parts.txt", "sha256": "b" * 64},
            ],
        },
        "tables": tables,
    }
    source_map = {}
    ordered_small = sorted(SMALL_INTESTINE_MEMBERS, key=lambda item: int(item[2:]))
    ordered_colon = sorted(COLON_COMPONENTS, key=lambda item: int(item[2:]))
    for stable_id, member in [
        *((398 + offset, member) for offset, member in enumerate(ordered_small)),
        *((454 + offset, member) for offset, member in enumerate(ordered_colon)),
    ]:
        small = member in SMALL_INTESTINE_MEMBERS
        family_id = "FMA7200" if small else "FMA7201"
        family_label = "small intestine" if small else "large intestine"
        digest = hashlib.sha256(member.encode()).hexdigest()
        source_map[str(stable_id)] = {
            "name": "ileocecal junction" if member == "FJ2599" else family_label,
            "source_member": member,
            "source_sha256": digest,
            "layer": 1,
            "body_index": 7,
            "source_owner_metadata": {
                "concept_id": family_id,
                "hierarchy": "part_of",
                "label": family_label,
                "member_id": member,
                "member_sha256": digest,
            },
        }
    for stable_id in (4, 5, 6, 7, 8, 9, 10, 11, 318, 319, 320, 321):
        source_map[str(stable_id)] = {"name": "other", "layer": 1}
    receipt = {
        "payload": {"path": "/retained/resting-thorax.nhanatomy", "sha256": "c" * 64},
        "functional_bindings": {"anatomy_payload_sha256": "c" * 64},
        "mass_geometry_accounting": {"reference_mass_kg": 72.0, "source_mass_kg": 97.13},
        "provenance": {"source_id_map": source_map},
    }
    return atlas, receipt


class PassiveVisceraIdentityTests(unittest.TestCase):
    def test_wall_component_keeps_family_blood_ownership(self):
        atlas = {"tables": {"part_of": {
            ("FMA15043", "taenia omentalis"): {"FJ2570"},
        }}}
        identity = anatomical_component_identity(atlas, "FJ2570")
        self.assertEqual(identity["geometry_role"], "colon wall muscle component")
        self.assertFalse(identity["separate_physiological_compartment"])
        # Existing compartment bindings use the parent family; enriching source
        # identity must not remove a member or create a second blood compartment.
        source = {str(i): {"name": "other", "layer": 1}
                  for i in (4, 5, 6, 7, 8, 9, 10, 11, 318, 319, 320, 321)}
        source["458"] = {"name": "large intestine", "layer": 1}
        before = vascular_bindings(source)
        source["458"]["source_owner_metadata"] = {"anatomical_component": identity}
        self.assertEqual(before, vascular_bindings(source))
        splanchnic = next(x for x in before if x["cvsim_compartment_id"] == 10)
        self.assertIn(458, splanchnic["stable_ids"])

    def test_changed_source_membership_is_rejected(self):
        atlas = {"tables": {"part_of": {
            ("FMA15043", "taenia omentalis"): {"FJ2566"},
        }}}
        with self.assertRaisesRegex(ValueError, "membership changed"):
            anatomical_component_identity(atlas, "FJ2570")

    def test_unknown_component_is_not_inferred(self):
        self.assertIsNone(anatomical_component_identity({}, "FJ99999"))

    def test_bowel_zones_and_junction_keep_specific_source_identity(self):
        atlas = {"tables": {"is_a": {
            ("FMA7206", "duodenum"): {"FJ2573"},
            ("FMA14964", "proximal part of ileum"): {"FJ2574"},
            ("FMA11338", "ileocecal junction"): {"FJ2599"},
            ("FMA16981", "proximal part of jejunum"): {"FJ2627"},
        }}}
        for member, role in (("FJ2573", "small intestine segment"),
                             ("FJ2574", "small intestine zone"),
                             ("FJ2599", "intestinal junction"),
                             ("FJ2627", "small intestine zone")):
            with self.subTest(member=member):
                identity = anatomical_component_identity(atlas, member)
                self.assertEqual(identity["geometry_role"], role)
                self.assertEqual(identity["family_concept_id"], "FMA7200")
                self.assertEqual(identity["hierarchy"], "is_a")
                self.assertFalse(identity["separate_physiological_compartment"])

    def test_missing_or_ambiguous_bowel_zone_is_rejected(self):
        for table in ({}, {
                ("FMA14964", "proximal part of ileum"): {"FJ2574"},
                ("FMA16981", "proximal part of jejunum"): {"FJ2574"}}):
            with self.subTest(table=table):
                with self.assertRaisesRegex(ValueError, "membership changed or ambiguous"):
                    anatomical_component_identity({"tables": {"is_a": table}}, "FJ2574")

    def test_receipt_refresh_adds_exact_source_identity_without_reimport(self):
        atlas, receipt = _refresh_fixture()
        original = deepcopy(receipt)
        refreshed = refresh_passive_component_identities(receipt, atlas)
        self.assertEqual(receipt, original)
        self.assertEqual(refreshed["payload"], original["payload"])
        self.assertEqual(refreshed["functional_bindings"], original["functional_bindings"])
        self.assertEqual(refreshed["mass_geometry_accounting"], original["mass_geometry_accounting"])
        self.assertEqual(set(refreshed["provenance"]["source_id_map"]),
                         set(original["provenance"]["source_id_map"]))
        for stable_id, old in original["provenance"]["source_id_map"].items():
            new = refreshed["provenance"]["source_id_map"][stable_id]
            if old.get("source_member") not in SMALL_INTESTINE_MEMBERS | frozenset(COLON_COMPONENTS):
                self.assertEqual(new, old)
                continue
            expected = anatomical_component_identity(atlas, old["source_member"])
            expected_owner = dict(old["source_owner_metadata"], anatomical_component=expected)
            self.assertEqual(new["name"], expected["name"])
            self.assertEqual(new["source_member"], old["source_member"])
            self.assertEqual(new["source_sha256"], old["source_sha256"])
            self.assertEqual(new["source_owner_metadata"], expected_owner)
            expected_family = ("small intestine" if old["source_member"] in SMALL_INTESTINE_MEMBERS
                               else "large intestine")
            self.assertEqual(new["source_owner_metadata"]["label"], expected_family)
        self.assertEqual(
            refreshed["provenance"]["source_id_map"]["424"]["source_owner_metadata"]["anatomical_component"]["concept_id"],
            "FMA11338",
        )
        self.assertEqual(
            refreshed["provenance"]["source_id_map"]["458"]["source_owner_metadata"]["anatomical_component"]["concept_id"],
            "FMA15043",
        )
        self.assertEqual(refreshed["provenance"]["source_id_map"]["398"]["name"], "duodenum")
        self.assertEqual(refreshed["provenance"]["source_id_map"]["454"]["name"], "ascending colon")
        self.assertEqual(refreshed["provenance"]["source_id_map"]["457"]["name"], "taenia mesocolica")
        self.assertEqual(refreshed["provenance"]["source_id_map"]["398"]["source_owner_metadata"]["label"],
                         "small intestine")
        self.assertEqual(refreshed["provenance"]["source_id_map"]["454"]["source_owner_metadata"]["label"],
                         "large intestine")
        before = vascular_bindings(original["provenance"]["source_id_map"])
        after = vascular_bindings(refreshed["provenance"]["source_id_map"])
        self.assertEqual(before, after)
        self.assertEqual(
            refresh_passive_component_identities(refreshed, atlas), refreshed,
            "refresh should be idempotent",
        )
        proof = refreshed["provenance"]["passive_component_identity_refresh"]
        self.assertEqual(proof["source_member_count"], 63)
        self.assertFalse(proof["geometry_changed"])
        self.assertEqual(proof["physical_mass_changed_kg"], 0)
        self.assertTrue(proof["source_map_component_display_names_changed"])
        self.assertFalse(proof["source_owner_family_labels_changed"])
        self.assertFalse(proof["vascular_region_memberships_changed"])

    def test_receipt_refresh_rejects_missing_duplicate_and_wrong_parent_members(self):
        atlas, receipt = _refresh_fixture()
        missing = deepcopy(receipt)
        del missing["provenance"]["source_id_map"]["398"]
        with self.assertRaisesRegex(ValueError, "incomplete or changed"):
            refresh_passive_component_identities(missing, atlas)

        duplicate = deepcopy(receipt)
        duplicate["provenance"]["source_id_map"]["999"] = deepcopy(
            duplicate["provenance"]["source_id_map"]["424"]
        )
        with self.assertRaisesRegex(ValueError, "duplicate rendered identities"):
            refresh_passive_component_identities(duplicate, atlas)

        wrong_parent = deepcopy(receipt)
        wrong_parent["provenance"]["source_id_map"]["424"]["source_owner_metadata"]["concept_id"] = "FMA11338"
        with self.assertRaisesRegex(ValueError, "registered family owner"):
            refresh_passive_component_identities(wrong_parent, atlas)

    def test_receipt_refresh_rejects_changed_table_provenance_or_component(self):
        atlas, receipt = _refresh_fixture()
        bad_tables = deepcopy(atlas)
        bad_tables["source"]["tables"] = bad_tables["source"]["tables"][:1]
        with self.assertRaisesRegex(ValueError, "both pinned source membership tables"):
            refresh_passive_component_identities(receipt, bad_tables)

        malformed_table = deepcopy(atlas)
        malformed_table["source"]["tables"][0]["file"] = []
        with self.assertRaisesRegex(ValueError, "malformed or duplicate"):
            refresh_passive_component_identities(receipt, malformed_table)

        malformed_receipt = deepcopy(receipt)
        malformed_receipt["payload"]["sha256"] = "z" * 64
        with self.assertRaisesRegex(ValueError, "payload identity"):
            refresh_passive_component_identities(malformed_receipt, atlas)

        bad_membership = deepcopy(atlas)
        bad_membership["tables"]["is_a"][("FMA14964", "proximal part of ileum")].remove("FJ2574")
        with self.assertRaisesRegex(ValueError, "membership changed or ambiguous"):
            refresh_passive_component_identities(receipt, bad_membership)


if __name__ == "__main__":
    unittest.main()
