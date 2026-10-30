import unittest, numpy as np
from verify_quality_proposal_034 import check_geometry, check_targets
class Tests(unittest.TestCase):
 def setUp(self):
  # Coplanar square with a redundant center. Retire center; leave oriented ring.
  self.s={'points_world_m':np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0],[.5,.5,0]],np.float64),'faces':np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])}
  self.p={'points_world_m':self.s['points_world_m'][:4].copy(),'faces':np.array([[0,1,2],[0,2,3]]),'point_output_to_input_rows':np.arange(4),'point_input_to_output_row':np.array([0,1,2,3,-1]),'face_output_to_input_rows':np.array([-1,-1]),'retired_input_face_rows':np.arange(4),'new_faces_input_point_rows':np.array([[0,1,2],[0,2,3]])}
 def target(self,row):
  return [{'role':'O','canonical_protected_row':3,'surface_vertex_row':row,'target_world_m_f32':self.s['points_world_m'][row].astype(np.float32).tolist()}]
 def test_target_row_compaction(self):
  self.assertEqual(check_targets('O',self.s,self.p,self.target(3))[0]['output_row'],3)
 def test_protected_retirement_rejected(self):
  with self.assertRaisesRegex(ValueError,'target retired'):check_targets('O',self.s,self.p,self.target(4))
 def test_stale_target_source_rejected(self):
  routes=self.target(3);routes[0]['surface_vertex_row']=0
  with self.assertRaisesRegex(ValueError,'source coordinate'):check_targets('O',self.s,self.p,routes)
 def test_unrelated_target_role(self):
  self.assertEqual(check_targets('I',self.s,self.p,self.target(4)),[])
 def test_valid_retirement(self):
  r,c=check_geometry(self.s,self.p);self.assertEqual((r['retired_points'],r['boundary_edges'],c.tolist()),(1,4,[0,1]))
 def test_coordinate_motion_rejected(self):
  self.p['points_world_m'][0,0]=1e-9
  with self.assertRaisesRegex(ValueError,'coordinates'):check_geometry(self.s,self.p)
 def test_silent_dtype_change_rejected(self):
  self.p['points_world_m']=self.p['points_world_m'].astype(np.float32)
  with self.assertRaisesRegex(ValueError,'dtype'):check_geometry(self.s,self.p)
 def test_bad_inverse_rejected(self):
  self.p['point_input_to_output_row'][4]=0
  with self.assertRaisesRegex(ValueError,'reciprocal'):check_geometry(self.s,self.p)
 def test_forged_retained_face_rejected(self):
  self.p['face_output_to_input_rows'][0]=0;self.p['retired_input_face_rows']=np.array([1,2,3])
  with self.assertRaisesRegex(ValueError,'connectivity'):check_geometry(self.s,self.p)
 def test_retired_partition_rejected(self):
  self.p['retired_input_face_rows']=np.array([0,1,1,3])
  with self.assertRaisesRegex(ValueError,'partition'):check_geometry(self.s,self.p)
 def test_orientation_rejected(self):
  self.p['faces'][1]=[0,3,2];self.p['new_faces_input_point_rows']=self.p['faces'].copy()
  with self.assertRaisesRegex(ValueError,'same-direction'):check_geometry(self.s,self.p)
 def test_unused_rows_rejected(self):
  self.p['points_world_m']=self.s['points_world_m'].copy()
  self.p['point_output_to_input_rows']=np.arange(5);self.p['point_input_to_output_row']=np.arange(5)
  with self.assertRaisesRegex(ValueError,'unused'):check_geometry(self.s,self.p)
 def test_float32_view_rejected(self):
  self.p['points_world_m_f32']=self.p['points_world_m'].astype(np.float32);self.p['points_world_m_f32'][0,2]=1e-4
  with self.assertRaisesRegex(ValueError,'Float32'):check_geometry(self.s,self.p)
if __name__=='__main__':unittest.main(verbosity=2)
