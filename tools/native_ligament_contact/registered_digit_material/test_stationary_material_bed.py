import unittest
import numpy as np
from fit_native_bed_barrier_load import force
from material_bed_release import stationary_bed,balance_emitted_mass

class MaterialBedTest(unittest.TestCase):
    def test_same_force_as_shipped_barrier_and_consistent_derivatives(self):
        w=1.e-5;dt=15.625e-6;g=np.array([.91*w,.985*w,1.2*w]);m=np.array([.005,.025,.002])
        e,f,k=stationary_bed(g,m,dt,w)
        np.testing.assert_allclose(f[:2],force(g[:2],m[:2],dt,w),rtol=4e-15)
        eps=1e-11
        plus=stationary_bed(g+eps,m,dt,w);minus=stationary_bed(g-eps,m,dt,w)
        np.testing.assert_allclose((plus[0]-minus[0])/(2*eps),-f,rtol=1e-8)
        np.testing.assert_allclose((plus[1]-minus[1])/(2*eps),-k,rtol=1e-8)
        self.assertEqual((e[-1],f[-1],k[-1]),(0.,0.,0.))
        self.assertTrue((k[:2]>0).all())

    def test_nonphysical_gap_is_rejected(self):
        with self.assertRaises(ValueError):
            stationary_bed(np.array([0.]),np.array([.01]),1e-4,1e-5)

    def test_rounding_honors_donors_and_contact_points(self):
        base=np.array([[.26,.2501,.7502],[.75,.7502,.6502],[.73,.31,.2]],dtype='f4').astype(float)
        old=base.copy()
        old[0,0]=np.nextafter(np.float32(old[0,0]),np.float32(1.0))
        W=np.array([[1.,.5,0.],[1.,0.,.5]])
        fixed=np.zeros((3,3),dtype=bool);fixed[0,:]=True
        with self.assertRaises(ValueError):
            balance_emitted_mass(base,old,np.array([[1.,.5,0.],[1.,0.,1.]]),[1e-9,1e-9],fixed)
        out=balance_emitted_mass(base,old,W,[1e-9,1e-9],fixed)
        np.testing.assert_array_equal(out[fixed],old[fixed])
        self.assertLess(np.max(abs(W@(out-base))),1e-9)
        np.testing.assert_array_equal(out,out.astype('f4'))

    def test_compiled_contact_membership_follows_emitted_coordinates(self):
        import tempfile
        from pathlib import Path
        from material_bed_release import bind_contact_footprint
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);data=d/'NumiHumanLoadedSkinPoseData.inc';data.write_text('namespace numiLoadedData {}')
            loader=d/'NumiHumanLoadedSkinPose.hpp'
            loader.write_text('    size_t touch=0;\n    require(bundle.boundaryMap.outer.size()==3,"bad");\n    require(touch==3,"no registered physical skin point touches the bed");')
            p=np.array([[0.,0.,.25],[0.,1.,.72],[1.,0.,.6]])
            self.assertEqual(bind_contact_footprint(d,[2,0,1],p,.25,.65),[0,2])
            self.assertIn('0u,2u',data.read_text())
            p[1,2]=.60;p[2,2]=.72
            self.assertEqual(bind_contact_footprint(d,[1,2,0],p,.25,.65),[0,1])
            self.assertIn('0u,1u',data.read_text());self.assertNotIn('0u,2u',data.read_text())
            self.assertEqual(loader.read_text().count('std::equal'),1)
            with self.assertRaises(ValueError):bind_contact_footprint(d,[0,0,1],p,.25,.65)
            with self.assertRaises(ValueError):bind_contact_footprint(d,[0,1,2],p,.30,.65)
