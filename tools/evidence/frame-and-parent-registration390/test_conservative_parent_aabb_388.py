from __future__ import annotations
import importlib.util,sys,time,unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
from conservative_parent_aabb_388 import parent_aabb_candidate_mask
P=Path(__file__).parent

def load(name,path):
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
old=load('parent_registration_314_reference',P/'bounded-parent-registration-314/export_authoring_registered_surface_sidecars_230.py')
new=load('parent_registration_388_filtered',P/'bounded-parent-registration-388/export_authoring_registered_surface_sidecars_230.py')

class ConservativeBoundsTests(unittest.TestCase):
    def assert_no_false_exclusion(self,point,low,high,upper):
        mask=parent_aabb_candidate_mask(point,low,high,upper)
        exact=np.array([old._exact_aabb_distance_squared(point,a,b)<=upper for a,b in zip(low,high)])
        self.assertFalse(np.any(exact&~mask))
    def test_random_scales_and_boundary_ulp(self):
        rng=np.random.default_rng(388)
        for exponent in (-150,-20,-1,0,1,20,150):
            scale=10.**exponent
            point=rng.normal(size=3)*scale
            low=rng.normal(size=(73,3))*scale
            high=low+np.abs(rng.normal(size=(73,3)))*scale
            low[0]=point;high[0]=np.nextafter(point,np.inf)
            for row in (0,1,17,72):
                upper=old._exact_aabb_distance_squared(point,low[row],high[row])
                self.assert_no_false_exclusion(point,low,high,upper)
    def test_underflow_overflow_and_zero(self):
        for magnitude in (0.,np.nextafter(0.,1.),1e-308,1e308):
            point=np.array([magnitude,0.,0.]);low=np.array([[-magnitude,0.,0.],[magnitude,0.,0.]])
            for upper in (Fraction(0),Fraction.from_float(magnitude)**2*4):
                self.assert_no_false_exclusion(point,low,low,upper)
    def test_empty_and_invalid(self):
        self.assertEqual(len(parent_aabb_candidate_mask(np.zeros(3),np.empty((0,3)),np.empty((0,3)),Fraction(0))),0)
        for low,high in [(np.zeros((1,2)),np.zeros((1,2))), (np.ones((1,3)),np.zeros((1,3))), (np.full((1,3),np.nan),np.zeros((1,3)))]:
            with self.assertRaises(ValueError):parent_aabb_candidate_mask(np.zeros(3),low,high,Fraction(0))
    def test_exact_projection_and_tie_parity(self):
        unit=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        vertices=np.concatenate([unit,unit+np.array([2.,0.,0.]),unit])
        tets=np.arange(12).reshape(3,4);points=vertices[tets]
        parent={'points':vertices,'tets':tets}
        mi={'globalRows':np.arange(3),'boundsMin':points.min(1),'boundsMax':points.max(1)}
        for point in (np.array([.2,.2,.2]),np.array([1.5,0.,0.]),np.array([.5,.5,.5]),np.array([10.,1.,1.])):
            outcomes=[]
            for owner in (old,new):
                try:
                    result=owner._exact_nearest_same_material(parent,mi,15,point,0)
                    outcomes.append((result['row'],result['q'],result['bary'],result['d2'],result['ties']))
                except owner.ExactProjectionTieError as error:
                    d=error.diagnostic;outcomes.append((d['status'],d['nearestDistanceSquaredM2Exact'],d['exactMinimizerParentTetRows'],d['exactNearestDistinctProjectionCount']))
            self.assertEqual(outcomes[0],outcomes[1])
    def test_measured_large_parent_filter(self):
        rng=np.random.default_rng(389);count=28632
        low=rng.uniform(-2.,2.,(count,3));high=low+.01
        point=np.array([0.,0.,0.]);upper=Fraction(1,10**8)
        started=time.perf_counter();mask=parent_aabb_candidate_mask(point,low,high,upper);filtered_s=time.perf_counter()-started
        started=time.perf_counter();exact=np.array([old._exact_aabb_distance_squared(point,a,b)<=upper for a,b in zip(low,high)]);reference_s=time.perf_counter()-started
        self.assertTrue(np.all(mask|~exact))
        print({'rows':count,'filter_wall_s':filtered_s,'reference_exact_box_wall_s':reference_s,'filter_survivors':int(mask.sum()),'exact_survivors':int(exact.sum()),'physical_admission':False})

if __name__=='__main__':unittest.main()
