import unittest
import numpy as np
from registered_cavity_materials import apply_registered_cavities


class CavityMaterials(unittest.TestCase):
    def setUp(self):
        self.cells = np.arange(20).reshape(5, 4)
        self.labels = np.array([15, 16, 15, 16, 15])
        self.cavity = dict(retained_input_cell_rows=np.array([4, 1]),
                           replacement_tetrahedra=np.array([[1, 2, 3, 4], [4, 3, 2, 1]]),
                           replacement_parent_offsets=np.array([0, 2, 3]),
                           replacement_parent_cell_rows=np.array([2, 0, 3]))

    def test_distinct_material_cavities_in_native_order(self):
        cells, labels = apply_registered_cavities(self.cells, self.labels, self.cavity)
        np.testing.assert_array_equal(labels, [15, 16, 15, 16])
        np.testing.assert_array_equal(cells[:2], self.cells[[4, 1]])
        np.testing.assert_array_equal(cells[2:], self.cavity["replacement_tetrahedra"])

    def test_invalid_material_ancestry(self):
        for key, values in (("replacement_parent_cell_rows", [3, 0, 2]),
                            ("replacement_parent_cell_rows", [2, 0, 4]),
                            ("replacement_parent_cell_rows", [2, 2, 3]),
                            ("replacement_parent_cell_rows", [-1, 0, 3]),
                            ("retained_input_cell_rows", [1, 1]),
                            ("replacement_parent_offsets", [0, 0, 3])):
            with self.subTest(key=key, values=values), self.assertRaises(ValueError):
                apply_registered_cavities(self.cells, self.labels,
                                          dict(self.cavity, **{key: np.array(values)}))
