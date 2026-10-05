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


if __name__ == "__main__":
    unittest.main()
