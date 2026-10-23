"""Regress stale thickness metadata, unknown bounds and metre/F32 emission."""
import unittest
import numpy as np
from numilab_human.skin_geometry_diagnostics import _skin_layer_reference_diagnostics

class SkinLayerReferenceDiagnosticsTests(unittest.TestCase):
    def fixture(self):
        # Far from the origin, emission order matters for submillimetre layers.
        o=np.array([[0., 0., 2.], [.1, 0., 2.]])
        i=o-np.array([0.,0.,.0006]);d=i-np.array([0.,0.,.0005])
        emitted=[x.astype('f4').astype(float) for x in (o,i,d)]
        skin=np.linalg.norm(emitted[0]-emitted[1],axis=1)
        fat=np.linalg.norm(emitted[1]-emitted[2],axis=1)
        return [o,i,d,skin,fat,np.full(2,.0005),np.full(2,.001),
                np.full(2,.0004),np.full(2,.001)]
    def test_native_width_identity_is_not_physical_admission(self):
        result=_skin_layer_reference_diagnostics(*self.fixture())
        self.assertTrue(result['reference_width_checks_pass'])
        self.assertFalse(result['physical_admission'])
        self.assertEqual(result['layers']['fat']['maximum_declared_width_error_m'],0.)
    def test_stale_thickness_only_repair_fails_identity(self):
        a=self.fixture();a[4][1]+=.000166
        result=_skin_layer_reference_diagnostics(*a)
        self.assertFalse(result['reference_width_checks_pass'])
        self.assertEqual(result['layers']['fat']['declared_width_mismatch_rows'],[1])
    def test_actual_geometry_must_clear_the_exact_authored_floor(self):
        a=self.fixture();a[7][0]=a[4][0]+1e-12
        result=_skin_layer_reference_diagnostics(*a)
        self.assertEqual(result['layers']['fat']['outside_authored_interval_rows'],[0])
        self.assertFalse(result['reference_width_checks_pass'])
    def test_unknown_midpoint_intervals_do_not_silently_pass(self):
        a=self.fixture();a[7][0]=0.;a[8][0]=np.inf
        result=_skin_layer_reference_diagnostics(*a)
        self.assertEqual(result['layers']['fat']['unknown_or_invalid_interval_rows'],[0])
        self.assertFalse(result['reference_width_checks_pass'])
    def test_cast_metres_once_instead_of_casting_scaled_millimetres(self):
        a=self.fixture();result=_skin_layer_reference_diagnostics(*a)
        actual=result['layers']['skin']['minimum_emitted_width_m']
        wrong=np.linalg.norm((a[0]*1000).astype('f4').astype(float)-
                             (a[1]*1000).astype('f4').astype(float),axis=1).min()/1000
        self.assertNotEqual(actual,wrong)
        self.assertAlmostEqual(actual,float(a[3].min()),places=15)
    def test_nonfinite_coordinates_and_invalid_intervals_reject(self):
        for column in (0,3,7):
            a=self.fixture();a[column].flat[0]=np.nan
            with self.subTest(column=column),self.assertRaises(ValueError):
                _skin_layer_reference_diagnostics(*a)
    def test_misaligned_rows_reject(self):
        a=self.fixture();a[1]=a[1][:1]
        with self.assertRaises(ValueError):_skin_layer_reference_diagnostics(*a)

if __name__=='__main__':unittest.main()
