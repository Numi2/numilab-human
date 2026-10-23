"""Source-frame and direct/projected identity regressions for joint refinement."""
import unittest
import numpy as np
from numilab_human.skin_geometry_diagnostics import _skin_source_embedding_diagnostics as check

class SourceEmbeddingDiagnosticsTests(unittest.TestCase):
    def fixture(self):
        x=np.array([[0.,0.,2.],[1.,0.,2.],[0.,1.,2.]])
        o=np.array([x[0],x[1],[.25,.25,2.]])
        return [x,np.array([[0,1,2]]),o,np.array([0,1,-1]),
                np.array([-1,-1,0]),np.array([[0.,0.,0.],[0.,0.,0.],[.5,.25,.25]]),np.zeros((3,3))]
    def test_direct_and_projected_identity_pass_without_admission(self):
        r=check(*self.fixture())
        self.assertTrue(r['source_embedding_checks_pass'])
        self.assertFalse(r['physical_admission'])
    def test_duplicate_direct_source_id_is_not_silently_deduplicated(self):
        a=self.fixture();a[3][1]=0;a[2][1]=a[2][0]
        r=check(*a)
        self.assertEqual(r['duplicate_direct_identity_rows'],[0,1])
        self.assertFalse(r['source_embedding_checks_pass'])
    def test_repeated_one_hot_sample_is_valid_projected_ancestry(self):
        a=self.fixture();a[3][1]=-1;a[4][1]=0;a[5][1]=[1,0,0];a[2][1]=a[2][0]
        self.assertTrue(check(*a)['source_embedding_checks_pass'])
    def test_direct_and_one_hot_identity_must_agree(self):
        a=self.fixture();a[4][1]=0;a[5][1]=[1,0,0]
        self.assertEqual(check(*a)['direct_embedding_identity_mismatch_rows'],[1])
    def test_raw_atlas_residual_cannot_replace_registered_world_residual(self):
        a=self.fixture();a[6][2,2]=.35
        self.assertEqual(check(*a)['registered_residual_mismatch_rows'],[2])
    def test_missing_face_and_bad_barycentrics_are_explicit_failures(self):
        a=self.fixture();a[4][2]=-1
        self.assertEqual(check(*a)['missing_source_identity_rows'],[2])
        a=self.fixture();a[5][2]=[1,1,1]
        self.assertEqual(check(*a)['invalid_barycentric_rows'],[2])
    def test_uint32_sentinel_and_displacement_cap(self):
        a=self.fixture();a[3]=a[3].astype(np.uint32);a[4]=a[4].astype(np.uint32)
        a[2][2,2]+=.001;a[6][2,2]=a[2].astype('f4').astype(float)[2,2]-2.
        r=check(*a,displacement_caps_m=np.full(3,.0005))
        self.assertEqual(r['outside_declared_displacement_cap_rows'],[2])
    def test_float_indices_and_out_of_range_indices_reject(self):
        a=self.fixture();a[3]=a[3].astype(float)
        with self.assertRaises(ValueError):check(*a)
        a=self.fixture();a[4][2]=2
        with self.assertRaises(ValueError):check(*a)

if __name__=='__main__':unittest.main()
