import unittest
from numilab_human.open_knee import _cartilage_material_values


class KneeCartilageMaterialTests(unittest.TestCase):
    def test_source_solid_parameters_are_retained(self):
        source = dict(type="Mooney-Rivlin", c1=2.54, c2=0., k=100.)
        for name in ("FMC", "PTC", "TBC-L", "TBC-M"):
            self.assertEqual(_cartilage_material_values(name, source), (2.54, 0., 100.))

    def test_no_literature_or_wrong_law_fallback(self):
        valid = dict(type="Mooney-Rivlin", c1=2.54, c2=0., k=100.)
        for change in ({"k": float("nan")}, {"c1": 0}, {"type": "biphasic"}, {"c2": -1}):
            with self.assertRaises(RuntimeError):
                _cartilage_material_values("PTC", valid | change)
        with self.assertRaises(RuntimeError):
            _cartilage_material_values("PTC", {})
        with self.assertRaises(RuntimeError):
            _cartilage_material_values("ACL", valid)
