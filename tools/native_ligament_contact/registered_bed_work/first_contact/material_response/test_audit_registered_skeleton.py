import unittest

import numpy as np

from audit_registered_skeleton import closed_solid, intersection_mm3


class SkeletonAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        self.faces = np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
        self.original = closed_solid(self.vertices, self.faces)

    def test_enclosed_bone_with_no_coincident_vertices(self):
        smaller = 0.2*self.vertices + 0.1
        overlap = intersection_mm3(self.original, closed_solid(smaller,self.faces))
        self.assertAlmostEqual(overlap, 0.2**3/6., places=12)

    def test_touching_cortex_and_duplicate_source_seams(self):
        # The tetrahedra share the z=0 triangle. Their interiors are disjoint.
        reflected = self.vertices.copy()
        reflected[:,2] *= -1
        split_points = reflected[self.faces].reshape(-1,3)
        split_faces = np.arange(12).reshape(-1,3)
        self.assertEqual(intersection_mm3(self.original, closed_solid(split_points, split_faces)), 0.)

    def test_open_cortical_shell_is_rejected(self):
        with self.assertRaises(ValueError):
            closed_solid(self.vertices,self.faces[:-1])


if __name__ == '__main__':
    unittest.main()
