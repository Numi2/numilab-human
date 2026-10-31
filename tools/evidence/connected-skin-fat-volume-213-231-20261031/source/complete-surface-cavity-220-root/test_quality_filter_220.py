import importlib.util
from pathlib import Path
import unittest
import tempfile
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
SCRIPT=H/"complete-surface-cavity-220-root/solve_oriented_cavity_milp_220.py"
SPEC=importlib.util.spec_from_file_location("cavity220",SCRIPT)
c220=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c220)

class NativeQualityFilterTests(unittest.TestCase):
    def test_filter_uses_both_unchanged_native_thresholds(self):
        points=np.asarray([
            [0,0,0],[1,0,0],[0,1,0],[0,0,1],
            [2,0,0],[3,0,0],[2,1,0],[2,0,1e-6],
            [4,0,0],[4+1e-6,0,0],[4,1e-6,0],[4,0,1e-6],
        ],dtype=np.float32)
        tets=np.asarray([[0,1,2,3],[4,5,6,7],[8,9,10,11]],dtype=np.int64)
        accepted,det,normalized=c220.native_quality_mask(points,tets)
        self.assertEqual(accepted.tolist(),[True,False,False])
        self.assertGreater(det[1]/6,1e-18)  # rejected by normalized determinant
        self.assertLess(normalized[1],1e-5)
        self.assertLessEqual(det[2]/6,1e-18) # rejected by absolute volume threshold
        self.assertGreaterEqual(normalized[2],1e-5)

    def test_negative_orientation_is_not_accepted_by_native_quality_filter(self):
        points=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
        good=np.asarray([[0,1,2,3]],dtype=np.int64)
        reversed_tet=good[:,[1,0,2,3]]
        accepted,det,normalized=c220.native_quality_mask(points,reversed_tet)
        self.assertFalse(bool(accepted[0]))
        self.assertLess(float(det[0]),0.0)
        self.assertLess(float(normalized[0]),0.0)

    def test_filter_precedes_constraint_build_and_exact_milp(self):
        # The fifth point creates several skinny 4-tuples. They must be removed
        # before the face-balance matrix is built, while the original tet stays feasible.
        points=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0.2,0.2,1e-7]],dtype=np.float32)
        boundary=c220.PAT.copy()
        report,tets=c220.solve(points,boundary,seconds=3,max_points=8,material=15)
        self.assertEqual(report['status'],'PASS_EXACT_ORIENTED_CAVITY_TETRAHEDRALIZATION')
        self.assertGreater(report['native_quality_rejected_candidates'],0)
        self.assertEqual(report['candidate_tetrahedra'],report['native_quality_admissible_candidates'])
        self.assertLess(report['candidate_tetrahedra'],report['generated_candidate_tetrahedra'])
        self.assertTrue(report['exact_boundary_preserved'])
        self.assertTrue(report['exact_volume_conserved'])
        self.assertEqual(report['exact_internal_face_crossings'],0)
        self.assertEqual(report['material_index'],15)
        self.assertEqual(len(tets),1)

    def test_proposal_writer_keeps_explicit_role_material(self):
        points=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
        tets=np.asarray([[0,1,2,3]],dtype=np.int64)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'proposal.npz'
            digest=c220.write_proposal(out,points,tets,c220.PAT,15)
            self.assertEqual(digest,c220.sha(out))
            with np.load(out,allow_pickle=False) as z:
                self.assertEqual(z['material_indices'].tolist(),[15])
                self.assertTrue(np.array_equal(z['points_world_m'],points))

    def test_filter_owner_is_the_unchanged_198_gate(self):
        owner=H/"fixed_boundary_volume_flip_198_root.py"
        self.assertTrue(owner.is_file())
        self.assertEqual(c220.sha(c220.QUALITY_OWNER_PATH),c220.sha(owner))
        self.assertEqual(c220.QUALITY_OWNER.passes.__code__.co_consts,
                         __import__("runpy").run_path(str(owner))["passes"].__code__.co_consts)

if __name__=="__main__":
    unittest.main()
