import unittest
import numpy as np
from audit_settled_material_placement import transport_material_nodes


class WorldMaterialFrame(unittest.TestCase):
    def test_loaded_world_frame(self):
        origin = np.array([0.0, 1.0, 0.25 - 0.01171875])
        opening = dict(position_m=origin, quaternion_xyzw=[0.0, 0.0, 0.0, 1.0])
        points = origin + np.array([[0.2, 0.0, 0.1], [0.1, -0.1, 0.125]])
        np.testing.assert_array_equal(
            transport_material_nodes(points, opening, opening), points.astype("f4")
        )
        settled = dict(
            position_m=[0.0, 1.0, 0.25], quaternion_xyzw=[0.0, 0.0, 1.0, 0.0]
        )
        expected = np.array([[-0.2, 1.0, 0.35], [-0.1, 1.1, 0.375]], dtype="f4")
        np.testing.assert_array_equal(
            transport_material_nodes(points, opening, settled), expected
        )


if __name__ == "__main__":
    unittest.main()
