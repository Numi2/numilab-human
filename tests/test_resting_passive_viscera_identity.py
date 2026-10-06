import unittest

from numilab_human.resting_passive_viscera import (
    anatomical_component_identity, vascular_bindings,
)


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


if __name__ == "__main__":
    unittest.main()
