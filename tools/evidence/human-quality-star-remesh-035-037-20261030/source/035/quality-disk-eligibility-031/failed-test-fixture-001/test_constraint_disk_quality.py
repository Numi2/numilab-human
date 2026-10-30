import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
OWNER=ROOT/"deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py"
sys.path.insert(0,str(OWNER.parent.parent))
import numilab_human.cardiac_cavity_intersections as pred
import constraint_disk_quality as new
spec=importlib.util.spec_from_file_location("frozen616",ROOT/"constraint-disk-triangulation-616/constraint_disk.py")
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def solve(mod,p,**kw):
 p=np.asarray(p,np.float64);p32=p.astype(np.float32)
 lattice=[pred.float32_point_lattice_key(row) for row in p32]
 return mod.collision_aware_dp_disk(list(range(len(p))),p,p32,set(),pred._prepare_surface_aabb([]),pred,lattice,[],1000000,**kw)
def alt(p,t):
 v=np.asarray(p,np.float32).astype(np.float64)[t];a=v[1]-v[0];b=v[2]-v[0]
 return np.linalg.norm(np.cross(a,b))/max(np.linalg.norm(a),np.linalg.norm(b),np.linalg.norm(a-b))
class Tests(unittest.TestCase):
 def test_default_preserves_frozen616_result(self):
  p=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
  a,b=solve(old,p),solve(new,p)
  self.assertEqual(a[0],b[0]);self.assertEqual(a[1],b[1])
  for k,v in a[2]["statistics"].items():self.assertEqual(v,b[2]["statistics"][k])
 def test_default_keeps_nonzero_sliver_but_optin_rejects(self):
  p=[[0,0,0],[1,0,0],[.5,1e-10,0]]
  self.assertEqual(len(solve(new,p)[0]),1)
  with self.assertRaisesRegex(ValueError,"no disk triangulation"):solve(new,p,minimum_altitude_m=20e-9)
 def test_quality_filter_retains_good_polygon_solution(self):
  p=[[0,0,0],[.5,-1e-10,0],[1,0,0],[.5,1,0]]
  tri,_,r=solve(new,p,minimum_altitude_m=20e-9,minimum_angle_degrees=1e-4)
  self.assertTrue(all(alt(p,t)>=20e-9 for t in tri))
  self.assertGreater(r["statistics"]["rejected_quality_triangles"],0)
  self.assertEqual(new.validate_disk(tri,list(range(4)),set())["euler_characteristic"],1)
 def test_angle_floor_is_active(self):
  p=[[0,0,0],[1,0,0],[.01,.01,0]]
  self.assertEqual(len(solve(new,p)[0]),1)
  with self.assertRaisesRegex(ValueError,"no disk triangulation"):solve(new,p,minimum_angle_degrees=1.)
 def test_invalid_thresholds_fail_closed(self):
  p=[[0,0,0],[1,0,0],[0,1,0]]
  for kw in [{"minimum_altitude_m":-1},{"minimum_altitude_m":float("nan")},{"minimum_altitude_m":float("inf")},{"minimum_angle_degrees":-1},{"minimum_angle_degrees":61},{"minimum_angle_degrees":float("nan")}]:
   with self.subTest(kw=kw),self.assertRaises(ValueError):solve(new,p,**kw)
 def test_input_points_and_frozen_module_unchanged(self):
  p=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]]);before=p.copy()
  solve(new,p,minimum_altitude_m=20e-9)
  self.assertTrue(np.array_equal(p,before))
  import hashlib
  self.assertEqual(hashlib.sha256((ROOT/"constraint-disk-triangulation-616/constraint_disk.py").read_bytes()).hexdigest(),"eab86c0a1060a1292db4a9a01e0a766be8739ea2fa2c575a5658082a453d87ea")
if __name__=="__main__":unittest.main(verbosity=2)
