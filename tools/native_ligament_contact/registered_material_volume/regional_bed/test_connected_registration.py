import unittest
import numpy as np
from connected_registration import connected_nodes


class MaterialNeighborhood(unittest.TestCase):
    def test_disconnected_tissue_can_occupy_the_same_spatial_window(self):
        p = np.array([[0.,0,0],[.1,0,0],[0,.1,0],[0,0,.1]])
        points = np.vstack((p, p + [.001,0,0]))
        cells = np.array([[0,1,2,3],[4,5,6,7]])
        np.testing.assert_array_equal(connected_nodes(points,cells,[0],.15), [0,1,2,3])
        np.testing.assert_array_equal(connected_nodes(points,cells,[0,4],0), [0,4])

    def test_radius_limits_the_accumulated_edge_length(self):
        pts = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1],[1,.9,.1],[1,.8,.1],[1,.9,.2]])
        cells = np.array([[0,1,2,3],[1,4,5,6]])
        self.assertNotIn(4,connected_nodes(pts,cells,[0],1.5))
        self.assertIn(4,connected_nodes(pts,cells,[0],2.0))

    def test_excluded_owner_cannot_bridge_two_allowed_regions(self):
        pts = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,.1],[1,.8,.1],[1,.9,.2]])
        cells = np.array([[0,1,2,3],[1,4,5,6]])
        eligible = np.array([True,False,True,True,True,True,True])
        self.assertNotIn(4,connected_nodes(pts,cells,[0],5,eligible=eligible))
        with self.assertRaises(ValueError):
            connected_nodes(pts,cells,[1],5,eligible=eligible)
