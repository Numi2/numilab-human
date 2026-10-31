import importlib.util
from pathlib import Path
import unittest
import numpy as np

SCRIPT=Path(__file__).with_name("audit_combined_current_surface_diff.py")
spec=importlib.util.spec_from_file_location("combined_surface_diff_closed_retirement",SCRIPT)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class ClosedComponentRetirementControls(unittest.TestCase):
    def setUp(self):
        # Outward boundary of the positively oriented tetrahedron [0,1,2,3].
        self.faces=np.asarray([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],dtype=np.int64)
        self.rows=[0,1,2,3]
        self.points=[0,1,2,3]

    def test_accepts_isolated_oriented_closed_k4(self):
        result=audit.validate_closed_component_retirement(self.faces,self.rows,self.points)
        self.assertEqual(result["closed_edge_count"],6)
        self.assertTrue(result["isolated_from_retained_faces"])

    def test_rejects_open_or_inconsistently_oriented_component(self):
        bad=self.faces.copy();bad[0]=bad[0][::-1]
        with self.assertRaisesRegex(ValueError,"oppositely oriented closed 2-manifold"):
            audit.validate_closed_component_retirement(bad,self.rows,self.points)

    def test_rejects_retired_points_reused_by_retained_face(self):
        retained=np.asarray([[0,1,4]],dtype=np.int64)
        with self.assertRaisesRegex(ValueError,"still used by retained"):
            audit.validate_closed_component_retirement(np.vstack((self.faces,retained)),self.rows,self.points)

    def test_rejects_wrong_component_cardinality(self):
        with self.assertRaisesRegex(ValueError,"exactly four unique"):
            audit.validate_closed_component_retirement(self.faces,self.rows[:3],self.points)

if __name__=="__main__":unittest.main(verbosity=2)
