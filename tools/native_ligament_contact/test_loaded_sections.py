import unittest
from recover_loaded_sections import recover

class LoadedSections(unittest.TestCase):
    def test_loaded_series_section(self):
        r=recover(.002,.008,1.e-6,9.e-6,-.001,100)
        self.assertEqual(r['outer_z_m'],0)
        self.assertAlmostEqual(r['skin_fat_interface_z_m'],.0019)
        self.assertAlmostEqual(r['deep_fat_z_m'],.009)
        self.assertAlmostEqual(r['elastic_reaction_N'],100)
        self.assertAlmostEqual(r['elastic_energy_J'],.05)

    def test_unloaded_gap(self):
        r=recover(.002,.008,1.e-6,9.e-6,.004,0)
        self.assertAlmostEqual(r['outer_z_m'],.004)
        self.assertAlmostEqual(r['deep_fat_z_m'],.014)
        self.assertEqual(r['elastic_reaction_N'],0)

    def test_exhausted_layer(self):
        with self.assertRaisesRegex(ValueError,'Nonpositive'):
            recover(.002,.008,1.e-6,9.e-6,-.009,900)

if __name__=='__main__':unittest.main()
