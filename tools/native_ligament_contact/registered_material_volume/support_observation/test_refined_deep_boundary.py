import unittest

import numpy as np

from refine_deep_fat_boundary import nodal_basis
from register_refined_deep_nodes import closest_tet


class RegisteredDeepRefinement(unittest.TestCase):
    def test_edge_followed_by_face_split_preserves_force_moment_and_work(self):
        rows=nodal_basis(3,[[0,1,3]],[.25],[[1,2,3,4]],[[.1,.4,.5]])
        weights=np.array([rows[4].get(k,0.) for k in range(3)])
        np.testing.assert_allclose(weights,[.375,.225,.4])
        positions=np.array([[1.,3,5],[3,-2,1],[-4,2,-1]])
        velocities=np.array([[.6,1,-3],[-2,5,3],[.1,0,2]])
        force=np.array([1.,-2.,3.])
        reactions=weights[:,None]*force
        np.testing.assert_allclose(reactions.sum(0),force)
        np.testing.assert_allclose(np.cross(positions,reactions).sum(0),np.cross(weights@positions,force))
        self.assertAlmostEqual((velocities*reactions).sum(),(weights@velocities)@force)

    def test_invalid_ancestry_is_rejected(self):
        for edge,fraction in [([0,3,3],.2),([0,0,3],.5),([0,1,3],1.2)]:
            with self.subTest(edge=edge,fraction=fraction), self.assertRaises(ValueError):
                nodal_basis(3,[edge],[fraction],[],[])
        with self.assertRaises(ValueError):
            nodal_basis(4,[[0,1,4]],[.5],[[4,2,3,5]],[[.3,.3,.4]])

    def test_closest_source_tet_reconstructs_inside_and_exterior_points(self):
        corners=np.vstack((np.zeros(3),np.eye(3)))
        for query,expected in [([.1,.2,.3],[.1,.2,.3]),([.5,.2,-.4],[.5,.2,0]),([1,1,1],[1/3]*3)]:
            dist2,bary=closest_tet(np.array(query),corners)
            np.testing.assert_allclose(bary@corners,expected,atol=1e-14)
            self.assertAlmostEqual(bary.sum(),1)
            self.assertTrue(np.min(bary)>=0)
            self.assertAlmostEqual(dist2,np.sum((np.asarray(query)-expected)**2))


if __name__=="__main__":
    unittest.main()
