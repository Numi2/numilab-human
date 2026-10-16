import unittest
import numpy as np
from numilab_human.surface_precision_retriangulation import (
    collapse_closed_component_edge_preserving_allocations as collapse,
)
from tests.test_resting_lung_edge_repair import RestingLungEdgeRepairTests

class SourceAllocationCollapseTests(unittest.TestCase):
    def setUp(self):
        vertices, self.faces = RestingLungEdgeRepairTests._octahedron()
        self.positions = vertices[:, :3].copy()

    def run_collapse(self, positions=None, faces=None, **kwargs):
        return collapse(
            self.positions if positions is None else positions,
            self.faces if faces is None else faces,
            identity=(51004,49), keep_vertex=2, remove_vertex=0,
            max_endpoint_displacement_m=kwargs.pop("bound",2e-5), **kwargs)

    def test_keeps_source_slots_and_reports_exact_face_lineage(self):
        p,f=self.positions.copy(),self.faces.copy()
        result,proof=self.run_collapse()
        np.testing.assert_array_equal(self.positions,p)
        np.testing.assert_array_equal(self.faces,f)
        self.assertEqual(result.shape,(6,3))
        self.assertNotIn(0,result)
        self.assertEqual(len(proof["kept_parent_face_origins"]),len(result))
        self.assertEqual(proof["source_component_count"],1)
        self.assertEqual(proof["candidate_component_count"],1)

    def test_independent_component_and_unused_allocations_remain_untouched(self):
        other=self.positions+np.float32(.001)
        positions=np.vstack((self.positions,other,[[8.,9.,10.]])).astype(np.float32)
        faces=np.vstack((self.faces,self.faces+6))
        result,proof=self.run_collapse(positions,faces)
        np.testing.assert_array_equal(result[-len(self.faces):],faces[-len(self.faces):])
        self.assertEqual(proof["source_component_count"],2)
        self.assertEqual(proof["candidate_component_count"],2)
        self.assertTrue(np.all(result<12))
        self.assertEqual(positions.shape,(13,3))

    def test_exact_coordinate_seam_alias_is_retired_without_merging_other_classes(self):
        positions=np.vstack((self.positions,self.positions[0])).astype(np.float32)
        faces=self.faces.copy();faces[0,faces[0]==0]=6
        result,proof=self.run_collapse(positions,faces)
        self.assertEqual(proof["retired_active_source_slots"],[0,6])
        self.assertNotIn(0,result)
        self.assertNotIn(6,result)

    def test_rejects_open_component_without_mutation(self):
        p,f=self.positions.copy(),self.faces.copy()
        with self.assertRaisesRegex(ValueError,"closed"):
            self.run_collapse(faces=self.faces[:-1])
        np.testing.assert_array_equal(self.positions,p)
        np.testing.assert_array_equal(self.faces,f)

    def test_rejects_displacement_guard_failure(self):
        with self.assertRaisesRegex(ValueError,"endpoint-displacement"):
            self.run_collapse(bound=1e-9)

    def test_rejects_unreferenced_or_noninteger_source_endpoint(self):
        for endpoint in (True,6):
            with self.subTest(endpoint=endpoint),self.assertRaises(ValueError):
                collapse(np.vstack((self.positions,[[1.,2.,3.]])).astype(np.float32),
                         self.faces,identity=(51004,49),keep_vertex=2,
                         remove_vertex=endpoint,max_endpoint_displacement_m=2e-5)

    def test_requires_an_explicit_positive_finite_bound(self):
        for bound in (None, True, 0, float("inf"), float("nan")):
            with self.subTest(bound=bound),self.assertRaisesRegex(ValueError,"explicit positive"):
                self.run_collapse(bound=bound)

    def test_rejects_float64_source_and_inexact_coordinate_alias(self):
        with self.assertRaisesRegex(ValueError,"Float32"):
            self.run_collapse(positions=self.positions.astype(float))
        p=self.positions.copy();p[0]=p[2]
        with self.assertRaises(ValueError):
            self.run_collapse(positions=p)

if __name__=="__main__":
    unittest.main()

